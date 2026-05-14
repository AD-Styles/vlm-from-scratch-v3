"""Mini-LLaVA v3 Gradio Demo for Hugging Face Spaces.

v2 demo 대비 v3 만의 추가 기능:
  1. 한국어 응답 — KoLLaVA 데이터로 학습 (catastrophic forgetting 해소)
  2. Slim adapter — 1045 MB → 8.28 MB (PEFT 표준 대비 −99.21%)
                    추론 시 image_token_row.safetensors 자동 패치
  3. OOD detector — CLIP image-text similarity + LLM first-token entropy
                    학습 분포 밖 이미지에 "잘 모르겠음" 신호

이 파일은 HF Spaces 빌드 환경에서 자동 실행됩니다.
첫 실행 시 가중치를 HF Hub (AD-Styles/mini-llava-v3) 에서 다운로드합니다 (~14 MB).
"""
from __future__ import annotations

import os
import time

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from src.dataset import encode_for_inference
from src.model import MiniLLaVA
from src.ood_detection import OODDetector

WEIGHT_REPO = "AD-Styles/mini-llava-v3"
WEIGHT_LOCAL = "checkpoints/v3_step1_korean"

DEMO_BANNER_MD = """
# 🖼️ Mini-LLaVA — v3 Demo (Korean + Slim Adapter + OOD)

> v2 의 3가지 한계를 정조준한 진화 버전. 모든 개선은 데이터로 검증.
>
> | 개선 | v2 | **v3 (이 데모)** |
> |---|---|---|
> | 다국어 | ❌ 영문 only (catastrophic forgetting) | ✅ **영문 + 한국어** |
> | LoRA adapter | 1045 MB | **8.28 MB (−99.21%)** |
> | OOD 처리 | 무조건 답변 (hallucination) | **"모름" 가능** (CLIP+entropy) |
>
> 📖 자세한 분석: **[GitHub README](https://github.com/AD-Styles/vlm-from-scratch-v3)**
"""

FOOTER_MD = """
---
🔗 [GitHub](https://github.com/AD-Styles/vlm-from-scratch-v3)
· 🤗 [Weights](https://huggingface.co/AD-Styles/mini-llava-v3)
· 🔁 [v2 baseline](https://github.com/AD-Styles/vlm-from-scratch)
· 김도윤 (AD-Styles) · 2026

> 💡 **CPU 환경**에서 동작하므로 응답에 5-15초 소요됩니다. 첫 응답은 모델 로드로 더 느립니다.
"""

EXAMPLES = [
    ["What is in this image?"],
    ["What animal is in this image?"],
    ["Describe this image briefly."],
    ["이 이미지에 무엇이 보이나요?"],
    ["이 이미지의 색상은 무엇인가요?"],
    ["이 이미지를 한 문장으로 설명해주세요."],
]


def ensure_weights() -> tuple[str, str]:
    """HF Hub 에서 v3 가중치 다운로드 (첫 실행 1회). 이후는 캐시 재사용."""
    if not os.path.exists(WEIGHT_LOCAL):
        print(f"[init] Downloading v3 weights from {WEIGHT_REPO} → {WEIGHT_LOCAL}")
        snapshot_download(
            repo_id=WEIGHT_REPO,
            local_dir=WEIGHT_LOCAL,
        )
        print(f"[init] Download complete.")
    else:
        print(f"[init] Weights already present at {WEIGHT_LOCAL}")

    projector = os.path.join(WEIGHT_LOCAL, "projector.pt")
    lora_slim = os.path.join(WEIGHT_LOCAL, "lora_adapter_slim")
    return projector, lora_slim


# ─── Eager init at module load (Spaces best practice) ──────────────
print("[init] Preparing v3 weights ...")
PROJECTOR_PATH, LORA_PATH = ensure_weights()

print("[init] Building MiniLLaVA + slim adapter (CPU) ...")
MODEL = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
MODEL.load_projector(PROJECTOR_PATH, map_location="cpu")
MODEL.load_lora_adapter(LORA_PATH)
MODEL.to("cpu")
MODEL.eval()
print("[init] MiniLLaVA ready.")

print("[init] Building OOD detector (CLIP-ViT-B/32 + 57 categories) ...")
DETECTOR = OODDetector(threshold=0.5, device="cpu")
print("[init] OOD detector ready.")


def _merge_inputs(image: Image.Image, question: str):
    """Image + question → merged_embeds + merged_mask (LLM.generate 입력)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = MODEL.image_processor(image, return_tensors="pt")["pixel_values"]
    input_ids, attn = encode_for_inference(MODEL.tokenizer, question)
    input_ids = input_ids.unsqueeze(0)
    attn = attn.unsqueeze(0)

    text_embeds = MODEL.llm.get_input_embeddings()(input_ids)
    image_embeds = MODEL.encode_image(pixel_values)
    merged_embeds, merged_mask, _ = MODEL._merge(
        text_embeds, attn, image_embeds, input_ids, labels=None
    )
    return merged_embeds, merged_mask


def predict(
    image: Image.Image | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_ood: bool,
    ood_threshold: float,
):
    if image is None:
        return "⚠️ 이미지를 먼저 업로드해 주세요.", "", ""
    if not question or not question.strip():
        return "⚠️ 질문을 입력해 주세요.", "", ""

    t0 = time.time()
    merged_embeds, merged_mask = _merge_inputs(image, question.strip())

    # generate (output_scores=True 로 첫 토큰 logits 확보 → OOD 신호)
    with torch.no_grad():
        out = MODEL.llm.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0.0,
            temperature=float(temperature),
            top_p=float(top_p),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=MODEL.tokenizer.pad_token_id,
            eos_token_id=MODEL.tokenizer.eos_token_id,
        )

    answer = MODEL.tokenizer.decode(out.sequences[0], skip_special_tokens=True).strip()

    # OOD 평가 (체크박스 활성 시)
    ood_md = ""
    if enable_ood and len(out.scores) > 0:
        DETECTOR.threshold = float(ood_threshold)
        first_logits = out.scores[0][0]
        ood_result = DETECTOR.score(image, first_logits=first_logits)

        verdict_emoji = "⚠️" if ood_result.is_ood else "✅"
        verdict_label = (
            "OOD (학습 분포 밖 — 답변 신뢰도 낮음)"
            if ood_result.is_ood
            else "In-Distribution (학습 분포 안 — 답변 정상 신뢰도)"
        )
        ood_md = (
            f"### 🛡️ OOD Detector\n"
            f"| 항목 | 값 |\n"
            f"|---|---|\n"
            f"| 판정 | {verdict_emoji} **{verdict_label}** |\n"
            f"| ood_score | **{ood_result.ood_score:.3f}** "
            f"(threshold {ood_threshold:.2f}, > 면 OOD) |\n"
            f"| CLIP max similarity | {ood_result.clip_max_sim:.3f} "
            f"(best match: '{ood_result.clip_match}') |\n"
            f"| LLM first-token entropy | "
            f"{ood_result.llm_entropy:.3f} nats |\n"
        )
        if ood_result.is_ood:
            answer = f"⚠️ (모델이 OOD 신호를 감지했습니다 — 답변 참고용)\n\n{answer}"

    elapsed = time.time() - t0
    meta = (
        f"⏱️ {elapsed:.2f}s · max_new={int(max_new_tokens)} "
        f"· T={temperature} · top_p={top_p} · adapter=slim (8.28 MB)"
    )
    return answer, ood_md, meta


with gr.Blocks(title="Mini-LLaVA v3 Demo") as demo:
    gr.Markdown(DEMO_BANNER_MD)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="🖼️ 이미지 업로드", height=380)
            question_in = gr.Textbox(
                label="❓ 질문 (한국어/영어 모두 가능)",
                placeholder="예: 이 이미지에 무엇이 보이나요?",
                lines=2,
            )
            gr.Examples(examples=EXAMPLES, inputs=[question_in], label="💡 예시 질문")

            with gr.Accordion("⚙️ 생성 옵션 (고급)", open=False):
                max_new_tokens = gr.Slider(
                    16, 512, value=128, step=16, label="max_new_tokens"
                )
                temperature = gr.Slider(
                    0.1, 1.5, value=0.7, step=0.05, label="temperature"
                )
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")

            with gr.Accordion("🛡️ OOD Detector (v3 신규)", open=True):
                enable_ood = gr.Checkbox(
                    value=True,
                    label="OOD 검출 활성화 (CLIP similarity + LLM entropy)",
                )
                ood_threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="ood_threshold (이 값 초과 → OOD 판정)",
                )

            submit_btn = gr.Button("🚀 응답 생성", variant="primary")

        with gr.Column(scale=1):
            answer_out = gr.Textbox(
                label="🤖 모델 응답", lines=8, interactive=False
            )
            ood_out = gr.Markdown("")
            meta_out = gr.Markdown("")

    submit_btn.click(
        fn=predict,
        inputs=[
            image_in, question_in, max_new_tokens, temperature, top_p,
            enable_ood, ood_threshold,
        ],
        outputs=[answer_out, ood_out, meta_out],
    )
    question_in.submit(
        fn=predict,
        inputs=[
            image_in, question_in, max_new_tokens, temperature, top_p,
            enable_ood, ood_threshold,
        ],
        outputs=[answer_out, ood_out, meta_out],
    )

    gr.Markdown(FOOTER_MD)


if __name__ == "__main__":
    demo.launch()
