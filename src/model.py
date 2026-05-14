"""MiniLLaVA — CLIP-ViT + MultiModalProjector + Qwen2.5 Causal LM.

LLaVA-1.5의 핵심 아키텍처를 직접 구현. HuggingFace의 LlavaForConditionalGeneration
같은 고수준 클래스를 사용하지 않고, 텍스트/이미지 임베딩 융합과 splice 로직을
저수준에서 직접 다룬다.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from .config import IGNORE_INDEX, IMAGE_TOKEN, LLM_MODEL, VISION_MODEL


class MultiModalProjector(nn.Module):
    """CLIP의 시각 특징을 LLM의 임베딩 공간으로 매핑하는 2-layer MLP.

    LLaVA-1.5의 'mlp2x_gelu' projector를 그대로 따른다.
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MiniLLaVA(nn.Module):
    """Vision-Language Model.

    - CLIP-ViT는 항상 frozen (강력한 사전학습 시각 표현 활용)
    - LLM은 기본 frozen (LLaVA Stage 1 alignment)
    - Stage 1: projector 만 학습 (1.49M params)
    - Stage 2 (use_lora=True): projector + LoRA on q/k/v/o 동시 학습 (총 3.66M params)
    """

    def __init__(
        self,
        vision_model_name: str = VISION_MODEL,
        llm_model_name: str = LLM_MODEL,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        untie_embeddings: bool = False,
    ):
        super().__init__()

        self.vision = CLIPVisionModel.from_pretrained(vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        # transformers 5.x 는 dtype=, 4.x 는 torch_dtype= — 둘 다 지원하기 위해 동적 분기
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name, dtype=torch_dtype
            )
        except TypeError:  # transformers 4.x fallback
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name, torch_dtype=torch_dtype
            )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # <image> 플레이스홀더 추가
        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_TOKEN]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # v3 bonus 실험: tie_word_embeddings 분리
        # Qwen2.5 는 기본 tie_word_embeddings=True → embed_tokens 와 lm_head 가
        # 같은 tensor. LoRA 학습 시 lm_head 의 gradient 가 embed_tokens 까지
        # 미세 영향 → PEFT 가 adapter 저장할 때 embed_tokens 까지 자동 포함 (1GB).
        # untie 하면 두 가중치가 독립 → adapter 에 embed_tokens 안 들어감 → slim.
        if untie_embeddings:
            self._untie_embeddings()

        vision_hidden = self.vision.config.hidden_size
        llm_hidden = self.llm.config.hidden_size
        self.projector = MultiModalProjector(vision_hidden, llm_hidden)
        # projector 의 dtype 을 LLM 과 일치시킴 (bf16 학습 시 mixed-dtype 충돌 방지)
        # 단 trainable 파라미터의 fp32 master weight 는 optimizer 가 별도 관리할 수 있으므로
        # 학습 안정성보다 dtype 일관성을 우선
        if torch_dtype != torch.float32:
            self.projector = self.projector.to(torch_dtype)

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def _untie_embeddings(self) -> None:
        """lm_head 가중치를 embed_tokens 와 분리 (별도 Parameter 로 clone).

        Qwen2.5 의 tie_word_embeddings=True 상태에서 lm_head.weight 와
        get_input_embeddings().weight 는 동일한 tensor 객체. clone() 으로
        독립 Parameter 를 만들고 config 도 False 로 설정.
        """
        input_emb = self.llm.get_input_embeddings()
        lm_head = self.llm.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError(
                "LLM 에 output_embeddings(lm_head) 가 없음 — untie 불가"
            )
        # 동일 tensor 인지 확인 (이미 untied 면 skip)
        if lm_head.weight.data_ptr() == input_emb.weight.data_ptr():
            lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        # config 동기화 — 이후 save/load 가 untied 로 동작
        self.llm.config.tie_word_embeddings = False
        # 검증
        assert lm_head.weight.data_ptr() != input_emb.weight.data_ptr(), (
            "untie 실패 — lm_head 와 embed_tokens 가 여전히 같은 tensor"
        )
        print("[model] tie_word_embeddings 분리 완료 (lm_head ⊥ embed_tokens)")

    # ──────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, N_patches, D_llm]. CLS 토큰 제외.

        bf16 학습 시 dataloader 의 fp32 pixel_values + vision/projector 의 bf16
        가중치가 충돌하지 않도록 양쪽 모두 dtype 정렬.
        """
        # 1) vision encoder dtype 에 맞춰 pixel_values 변환
        vision_dtype = next(self.vision.parameters()).dtype
        if pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(vision_dtype)
        outputs = self.vision(pixel_values=pixel_values)
        patch_features = outputs.last_hidden_state[:, 1:, :]
        # 2) projector dtype 에 맞춰 patch_features 변환 (CLIP 이 fp32 로 promote 하는 경우 대비)
        proj_dtype = next(self.projector.parameters()).dtype
        if patch_features.dtype != proj_dtype:
            patch_features = patch_features.to(proj_dtype)
        return self.projector(patch_features)

    # ──────────────────────────────────────────────────────────────────
    # Embedding fusion: <image> 위치를 patch tokens로 splice
    # ──────────────────────────────────────────────────────────────────
    def _merge(
        self,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """input_ids에서 <image> 위치를 image_embeds(N개 patch)로 교체.

        - 모든 샘플은 정확히 1개의 <image> 토큰을 가진다고 가정
        - text/mask/label을 모두 일관되게 재정렬
        """
        B, L, D = text_embeds.shape
        N = image_embeds.shape[1]
        new_L = L - 1 + N

        device = text_embeds.device
        merged_embeds = torch.zeros(B, new_L, D, dtype=text_embeds.dtype, device=device)
        merged_mask = torch.zeros(B, new_L, dtype=attention_mask.dtype, device=device)
        merged_labels = (
            torch.full((B, new_L), IGNORE_INDEX, dtype=torch.long, device=device)
            if labels is not None
            else None
        )

        for b in range(B):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) != 1:
                raise ValueError(
                    f"sample {b}는 <image> 토큰이 {len(img_pos)}개 — 정확히 1개여야 합니다."
                )
            p = img_pos.item()

            # 앞 / 이미지 / 뒤 순으로 splice
            merged_embeds[b, :p] = text_embeds[b, :p]
            merged_embeds[b, p : p + N] = image_embeds[b]
            merged_embeds[b, p + N :] = text_embeds[b, p + 1 :]

            merged_mask[b, :p] = attention_mask[b, :p]
            merged_mask[b, p : p + N] = 1
            merged_mask[b, p + N :] = attention_mask[b, p + 1 :]

            if labels is not None:
                merged_labels[b, :p] = labels[b, :p]
                # 이미지 patch 위치는 IGNORE_INDEX 유지 (이미 채워둠)
                merged_labels[b, p + N :] = labels[b, p + 1 :]

        return merged_embeds, merged_mask, merged_labels

    # ──────────────────────────────────────────────────────────────────
    # Forward (학습)
    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)

        merged_embeds, merged_mask, merged_labels = self._merge(
            text_embeds, attention_mask, image_embeds, input_ids, labels
        )

        return self.llm(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            return_dict=True,
        )

    # ──────────────────────────────────────────────────────────────────
    # Generation (추론)
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)
        merged_embeds, merged_mask, _ = self._merge(
            text_embeds, attention_mask, image_embeds, input_ids, labels=None
        )

        return self.llm.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint I/O — projector만 저장 (LLM/CLIP은 HF에서 다시 로드)
    # ──────────────────────────────────────────────────────────────────
    def save_projector(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.projector.load_state_dict(state)

    def load_lora_adapter(self, adapter_path: str) -> None:
        """학습된 LoRA adapter를 frozen LLM 위에 부착."""
        from peft import PeftModel

        self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
        self.llm.eval()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
