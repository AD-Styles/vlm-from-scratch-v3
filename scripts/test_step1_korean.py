"""v3 Step 1 (Korean mix data + LoRA Stage 2) 결과 검증.

Test A: 영문 sanity — LoRA stage 2 가 영문 능력 보존했는지
Test B: 한국어 catastrophic forgetting 해소 — v2 의 미해결 문제

사용 (33 디렉토리에서):
  python scripts/test_step1_korean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/ 에서 from src.X import 가 동작하도록 부모 디렉토리를 path 에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.config import GenerationConfig  # noqa: E402
from src.infer import VLMInference  # noqa: E402


CKPT = "checkpoints/v3_step1_korean/projector.pt"
LORA = "checkpoints/v3_step1_korean/lora_adapter"
IMG = "assets/source_dog.jpg"


def main() -> None:
    print("=" * 60)
    print("v3 Step 1 (Korean mix data + LoRA) 결과 검증")
    print("=" * 60)

    img_path = Path(IMG)
    if not img_path.exists():
        raise FileNotFoundError(f"이미지 없음: {IMG}")
    img = Image.open(IMG)
    print(f"[image] {IMG} ({img.size[0]}x{img.size[1]} {img.mode})")

    print(f"\n[init] VLMInference 로드 중 ...")
    print(f"  projector: {CKPT}")
    print(f"  lora     : {LORA}")
    infer = VLMInference(checkpoint_path=CKPT, lora_adapter_path=LORA)
    print(f"[init] 로드 완료\n")

    gen_cfg = GenerationConfig(
        max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True
    )

    # ────────── Test A — 영문 sanity ──────────
    print("=" * 60)
    print("Test A — 영문 sanity (LoRA stage 2 가 영문 능력 보존?)")
    print("=" * 60)
    q_en = "Describe this image briefly."
    print(f"Q: {q_en}")
    r_a = infer(img, q_en, gen_cfg=gen_cfg)
    print(f"A: {r_a['answer']}")
    print(f"   (elapsed: {r_a['elapsed']:.2f}s)")

    # ────────── Test B — 한국어 catastrophic forgetting 해소 ──────────
    print("\n" + "=" * 60)
    print("Test B — 한국어 catastrophic forgetting 해소 검증")
    print("=" * 60)
    q_ko = "이 이미지에 무엇이 보이나요?"
    print(f"Q: {q_ko}")
    r_b = infer(img, q_ko, gen_cfg=gen_cfg)
    print(f"A: {r_b['answer']}")
    print(f"   (elapsed: {r_b['elapsed']:.2f}s)")

    print("\n" + "=" * 60)
    print("판정 가이드")
    print("=" * 60)
    print("  Test A: 영어로 개를 묘사하면 ✅ (LoRA stage 2 정상)")
    print("  Test B: 한국어로 (의미 있게) 답변하면 ✅ (forgetting 해소)")
    print("           영어로 답하거나 횡설수설하면 ❌ (forgetting 잔재)")
    print("=" * 60)


if __name__ == "__main__":
    main()
