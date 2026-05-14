---
license: mit
language:
  - en
  - ko
library_name: peft
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - clip
  - qwen2.5
  - lora
  - peft
  - llava
  - korean
  - ood-detection
  - mini-llava
---

# Mini-LLaVA v3 — Korean Multilingual + Slim LoRA + OOD Detection

> v2 의 미해결 과제 3가지 (한국어 forgetting, 1 GB adapter, OOD hallucination) 를 정조준한 진화 버전.
> CLIP-ViT-B/32 + MLP Projector + Qwen2.5-0.5B + LoRA(r=16) 를 직접 구현한 Vision-Language Model 의 학습 가중치.

## 📦 이 레포의 구성 (~14 MB total)

```
projector.pt                       5.7 MB   ← MultiModalProjector (CLIP→LLM 매핑)
lora_adapter_slim/
├─ adapter_config.json             1.1 KB   ← PEFT config (modules_to_save=None)
├─ adapter_model.safetensors       8.27 MB  ← LoRA weights (q/k/v/o, r=16)
├─ image_token_row.safetensors     7.17 KB  ← <image> 토큰 1 row 만 (slim 핵심)
└─ README.md (PEFT auto-generated)
```

**v2 대비 −99.21%** (1045 MB → 8.28 MB) — slim 화 원리는 [GitHub README §Slim Adapter](https://github.com/AD-Styles/vlm-from-scratch-v3#2%EF%B8%8F%E2%83%A3-slim-adapter--1045-mb--828-mb-%EC%9E%AC%ED%95%99%EC%8A%B5-0) 참조.

## 🚀 Quick Start

```python
import torch
from PIL import Image
from huggingface_hub import snapshot_download

# 1) v3 src 코드 가져오기 (GitHub)
#    git clone https://github.com/AD-Styles/vlm-from-scratch-v3
#    cd vlm-from-scratch-v3
from src.model import MiniLLaVA
from src.dataset import encode_for_inference
from src.ood_detection import OODDetector

# 2) 가중치 다운로드
local_dir = snapshot_download("AD-Styles/mini-llava-v3", local_dir="checkpoints/v3_step1_korean")

# 3) 모델 로드 (slim adapter 자동 인식)
model = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
model.load_projector(f"{local_dir}/projector.pt", map_location="cpu")
model.load_lora_adapter(f"{local_dir}/lora_adapter_slim")
model.to("cpu").eval()

# 4) 추론
image = Image.open("path/to/image.jpg").convert("RGB")
input_ids, attn = encode_for_inference(model.tokenizer, "이 이미지에 무엇이 보이나요?")
pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"]
with torch.no_grad():
    out = model.generate(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attn.unsqueeze(0),
        pixel_values=pixel_values,
        max_new_tokens=128,
    )
print(model.tokenizer.decode(out[0], skip_special_tokens=True))

# 5) (선택) OOD 검출
detector = OODDetector(threshold=0.5, device="cpu")
# generate 할 때 output_scores=True 로 first_logits 받아서 detector.score(image, first_logits) 호출
```

## ✨ v2 → v3 핵심 개선

| 항목 | v2 | **v3 (이 레포)** |
|---|---|---|
| 다국어 응답 | ❌ 영문 only (catastrophic forgetting) | ✅ **영문 + 한국어** |
| LoRA adapter | 1045 MB | **8.28 MB (−99.21%)** |
| OOD 처리 | 무조건 답변 (hallucination) | **"잘 모르겠음" 가능** (CLIP+entropy) |
| 다운로드 자산 총합 | ~1051 MB | **~14 MB** |

## 🧠 학습 데이터 (Step 1, 175분)

| Source | Sample 수 | 언어 |
|---|---|---|
| VQAv2 | 3K | 영문 |
| LocalizedNarratives | 3K | 영문 |
| A-OKVQA | 3K | 영문 |
| **KoLLaVA** (LLaVA-Instruct DeepL 한역) | **4K** | **한국어** |
| **합계** | **13K** | **Korean ratio 30.8%** |

## 🛡️ OOD Detector (선택)

```
ood_score = 0.6 × clip_signal + 0.4 × entropy_signal
is_ood    = ood_score > 0.5  (default)

clip_signal:    1 - max(CLIP-ViT-B/32 similarity to 57 in-dist categories)
entropy_signal: H(LLM first-token logits) / 8.0 nats
```

검증 결과 (`scripts/test_ood_integration.py`): In-Dist (실제 개) 0.365 (✅) · OOD (Pikachu 카툰) 0.505 (⚠️)

## 🪶 Slim Adapter — 핵심 기술

PEFT 표준은 `modules_to_save` (embed_tokens + lm_head) 을 **통째로** 저장 → 1 GB.
하지만 사전 분석으로 발견:

```
saved embed_tokens vs base Qwen2.5:
  첫 151,665 행: max diff = 0.000000e+00  (정확히 일치)
  마지막 1 행 (<image> 토큰): 학습된 representation
```

→ `image_token_row.safetensors` (7 KB) 만 별도 저장하고, 추론 시 base Qwen2.5 의 마지막 row 만 patch.
→ **greedy decoding 7/7 응답 비트 단위 일치** (`scripts/verify_slim_adapter.py`).

## ⚠️ 한계

- **0.5B LLM** — 이미지 내용 정확도는 여전히 한계 (개를 소로 오인 등)
- **CLIP-ViT-B/32** — 49 patches, ViT-L/14 ablation 진행했으나 효과 한계 → 미채택
- **57 OOD 카테고리** — COCO + 일상 객체 위주, 도메인 확장 시 카테고리 보강 권장

## 🔗 링크

- 📂 **Code**: [github.com/AD-Styles/vlm-from-scratch-v3](https://github.com/AD-Styles/vlm-from-scratch-v3)
- 🚀 **Live Demo**: [HF Spaces — mini-llava-v3-demo](https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo)
- 🔁 **v2 baseline**: [github.com/AD-Styles/vlm-from-scratch](https://github.com/AD-Styles/vlm-from-scratch)
- 🤗 **v2 weights**: [AD-Styles/mini-llava-stage2](https://huggingface.co/AD-Styles/mini-llava-stage2)
- 🚢 **Triton/vLLM deploy**: [github.com/AD-Styles/nlp-triton-deployment](https://github.com/AD-Styles/nlp-triton-deployment)

## 📜 License

MIT — © 2026 김도윤 (AD-Styles)

## 📚 Citation

```bibtex
@misc{kim2026minillavav3,
  title  = {Mini-LLaVA v3: Korean Multilingual + Slim LoRA Adapter + OOD Detection},
  author = {Kim, Doyun},
  year   = {2026},
  url    = {https://github.com/AD-Styles/vlm-from-scratch-v3}
}
```
