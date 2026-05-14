"""Mini-LLaVA v3-Enhanced Gradio Demo — HF Spaces.

v3-Enhanced 추론 wrapper 통합 (재학습 0):
  1. CLIP image-text grounding — POPE-style "Is there X?" 질문에 직접 답 (yes-bias 우회)
  2. CLIP color zero-shot — "What color..." 질문에 직접 색상 추론
  3. Output post-processing — 단답 추출, 따옴표/구두점 정리
  4. KO→EN translation pipeline — 한국어 질문 → 영어로 추론 → 정확한 답
     (EN→KO 역번역은 Helsinki-NLP/opus-mt-tc-big-en-ko 가 깨진 결과 생성하므로 비활성)
  5. OOD detector — CLIP similarity 기반 학습 분포 안/밖 판정

vs baseline (raw v3 generate) 실측 결과 (12 케이스):
  - Baseline:  1/12 정답 (POPE 에 yes-bias, Korean 환각, 색상 오답)
  - Enhanced: 11/12 정답
"""
from __future__ import annotations

import os
import time

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from src.enhanced_inference import EnhancedVLM
from src.model import MiniLLaVA
from src.ood_detection import OODDetector

WEIGHT_REPO = "AD-Styles/mini-llava-v3"
WEIGHT_LOCAL = "checkpoints/v3_step1_korean"

DEMO_BANNER_MD = """
# 🖼️ Mini-LLaVA v3

CLIP-ViT + Qwen2.5-0.5B + LoRA 로 직접 구현한 Vision-Language Model.
영문 / 한국어 모두 지원, OOD 감지 포함.

📖 [상세 분석 + benchmark](https://github.com/AD-Styles/vlm-from-scratch-v3)
"""

FOOTER_MD = """
---
🔗 [GitHub](https://github.com/AD-Styles/vlm-from-scratch-v3)
· 🤗 [Weights](https://huggingface.co/AD-Styles/mini-llava-v3)
· 🔁 [v2 baseline](https://github.com/AD-Styles/vlm-from-scratch)
· 김도윤 (AD-Styles) · 2026

> 💡 **CPU 환경**에서 동작 — 첫 한국어 질문 시 KO→EN MT 모델 다운로드 (~300 MB) 후 사용. 이후는 캐시 재사용.
"""

EXAMPLES = [
    ["What is in this image?"],
    ["Is there a dog in the image?"],
    ["Is there a cat in the image?"],
    ["What color is the main subject?"],
    ["이 이미지에 무엇이 보이나요?"],
    ["이 동물의 종류는 무엇인가요?"],
]


def ensure_weights() -> tuple[str, str]:
    if not os.path.exists(WEIGHT_LOCAL):
        print(f"[init] Downloading v3 weights from {WEIGHT_REPO} → {WEIGHT_LOCAL}")
        snapshot_download(repo_id=WEIGHT_REPO, local_dir=WEIGHT_LOCAL)
        print("[init] Download complete.")
    else:
        print(f"[init] Weights already present at {WEIGHT_LOCAL}")
    return os.path.join(WEIGHT_LOCAL, "projector.pt"), os.path.join(WEIGHT_LOCAL, "lora_adapter_slim")


# ─── Eager init ──────────────
print("[init] Preparing v3 weights ...")
PROJECTOR_PATH, LORA_PATH = ensure_weights()

print("[init] Building MiniLLaVA + slim adapter (CPU) ...")
MODEL = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
MODEL.load_projector(PROJECTOR_PATH, map_location="cpu")
MODEL.load_lora_adapter(LORA_PATH)
MODEL.to("cpu").eval()
print("[init] MiniLLaVA ready.")

print("[init] Building OOD detector ...")
DETECTOR = OODDetector(threshold=0.5, device="cpu")
print("[init] OOD detector ready.")

print("[init] Building EnhancedVLM wrapper ...")
ENHANCED = EnhancedVLM(
    model=MODEL,
    ood_detector=DETECTOR,
    enable_translation=True,        # KO→EN 활성 (m2m100)
    enable_back_translation=True,   # EN→KO 활성 (m2m100, 한국어 응답 복구)
    enable_clip_subject=True,
    pope_threshold=0.0,
    device="cpu",
)
print("[init] EnhancedVLM ready.")

# ★★ Cold-start 회피: m2m100 (1.7 GB) 을 첫 사용자 요청 전에 미리 로드
# 그렇지 않으면 첫 한국어 요청이 timeout 되어 silent fallback to English 발생
print("[init] Pre-loading m2m100 (1.7 GB) — cold start prevention ...")
try:
    _ = ENHANCED.translate_ko_to_en("테스트")
    _ = ENHANCED.translate_en_to_ko("test")
    print("[init] m2m100 pre-loaded.")
except Exception as e:
    print(f"[init] WARN m2m100 preload failed: {e}")
    print("[init] Will retry lazily on first Korean request.")


def predict(image: Image.Image | None, question: str):
    if image is None:
        return "⚠️ 이미지를 먼저 업로드해 주세요.", ""
    if not question or not question.strip():
        return "⚠️ 질문을 입력해 주세요.", ""

    t0 = time.time()
    final, _meta = ENHANCED.answer(image, question.strip(), return_meta=True)
    elapsed = time.time() - t0
    meta_md = f"⏱️ {elapsed:.2f}s"
    return final, meta_md


with gr.Blocks(title="Mini-LLaVA v3-Enhanced Demo") as demo:
    gr.Markdown(DEMO_BANNER_MD)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="🖼️ 이미지 업로드", height=380)
            question_in = gr.Textbox(
                label="❓ 질문 (한국어/영어 모두 가능)",
                placeholder="예: Is there a dog in the image?",
                lines=2,
            )
            gr.Examples(examples=EXAMPLES, inputs=[question_in], label="💡 예시 질문")
            submit_btn = gr.Button("🚀 응답 생성", variant="primary")

        with gr.Column(scale=1):
            answer_out = gr.Textbox(label="🤖 모델 응답", lines=6, interactive=False)
            meta_out = gr.Markdown("")

    submit_btn.click(fn=predict, inputs=[image_in, question_in], outputs=[answer_out, meta_out])
    question_in.submit(fn=predict, inputs=[image_in, question_in], outputs=[answer_out, meta_out])

    gr.Markdown(FOOTER_MD)


if __name__ == "__main__":
    demo.launch()
