---
title: Mini-LLaVA v3 Demo (Korean + Slim + OOD)
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.14.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: VLM from scratch — v3 (KR + 8.28MB adapter + OOD detector)
---

# Mini-LLaVA — v3 Demo

CLIP-ViT-B/32 + MLP Projector + Qwen2.5-0.5B (+ slim LoRA r=16) + OOD layer 로 직접 구현한 Vision-Language Model 의 데모 배포.
HuggingFace 의 `LlavaForConditionalGeneration` 같은 고수준 추상화 미사용 — `<image>` 토큰 splice 와 융합 로직 직접 구현.

## ✨ v3 의 3가지 검증된 개선 (vs v2 demo)

| 개선 | v2 demo | **v3 (이 데모)** |
|---|---|---|
| 다국어 | ❌ 영문 only (catastrophic forgetting) | ✅ **영문 + 한국어** (KoLLaVA 30.8% mix 학습) |
| LoRA adapter 크기 | 1045 MB | **8.28 MB (−99.21%)** |
| OOD 신호 | 무조건 답변 (hallucination) | **CLIP+entropy 기반 "모름" 가능** |
| 다운로드 자산 총합 | ~1051 MB | **~14 MB** |

## 🛡️ OOD Detector — v3 신규 layer

```
ood_score = 0.6 × clip_signal + 0.4 × entropy_signal

clip_signal  : 1 - max(CLIP similarity to 57 in-dist categories), normalized
entropy_signal: H(LLM first-token logits) / 8.0 nats, clipped [0, 1]

is_ood = ood_score > threshold (default 0.5)
```

검증 (`scripts/test_ood_integration.py`): In-Dist (실제 개) → ood_score 0.365 (OK ✅) · OOD (Pikachu 카툰) → ood_score 0.505 (OOD ⚠️)

## 🪶 Slim Adapter — 1045 MB → 8.28 MB

PEFT 표준은 `modules_to_save` (embed_tokens + lm_head) 을 통째로 저장 → 1 GB.
사전 분석으로 발견: 학습된 부분은 `<image>` 토큰 1 row 뿐 (151,665/151,666 행은 base Qwen2.5 와 100% 일치).

→ `image_token_row.safetensors` (7 KB) 만 별도 저장하고, 추론 시 base 의 마지막 row 만 patch.
→ greedy decoding 7/7 응답 비트 단위 일치 (`scripts/verify_slim_adapter.py`).

## 🔗 링크

- 📂 [Code (GitHub) — v3](https://github.com/AD-Styles/vlm-from-scratch-v3)
- 🤗 [Weights (HF Hub) — mini-llava-v3](https://huggingface.co/AD-Styles/mini-llava-v3)
- 🔁 [v2 baseline (GitHub)](https://github.com/AD-Styles/vlm-from-scratch)
- 🚢 [Triton/vLLM deploy 분리 레포](https://github.com/AD-Styles/nlp-triton-deployment)

## 💡 사용 팁

- **CPU 환경** 이라 응답에 5-15초 소요됩니다 (첫 응답은 모델 로드 추가).
- **한국어/영어 모두** 자연스럽게 응답 (v2 와 차별화 포인트).
- **OOD Detector 패널** 에서 모델의 신뢰도 신호 확인 (만화·추상화·낙서 등 입력 시 OOD ↑).
- 예시 질문 클릭 → 이미지 업로드 → "응답 생성".
- 이미지 내용 정확도는 0.5B LLM 한계 (개를 소로 오인 등) — [v3 README §한계](https://github.com/AD-Styles/vlm-from-scratch-v3#%EF%B8%8F-%ED%95%9C%EA%B3%84-limitations) 참조.

---
🤖 김도윤 (AD-Styles) · 2026 · MIT License
