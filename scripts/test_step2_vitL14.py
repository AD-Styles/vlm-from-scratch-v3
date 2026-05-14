"""v3 Step 2 (ViT-L/14 + LoRA Stage 2) 결과 검증.

Step 1 (ViT-B/32) 와 동일한 이미지/질문으로 비교:
- Test A: 영문 sanity — Step 1 에서 dog → cat 오인
- Test B: 한국어 catastrophic forgetting — Step 1 에서 cake/rocks hallucination
- Test C-EN/KO: Pikachu (OOD-ish) — vision encoder 능력 추가 검증

사용 (33 디렉토리에서):
  python scripts/test_step2_vitL14.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import VISION_MODEL_L14, GenerationConfig  # noqa: E402
from src.infer import VLMInference  # noqa: E402


CKPT = "checkpoints/v3_step2_stage2/projector.pt"
LORA = "checkpoints/v3_step2_stage2/lora_adapter"
IMG_DOG = "assets/source_dog.jpg"
IMG_PIKA = "assets/source_pikachu.png"


def main() -> None:
    print("=" * 72)
    print("v3 Step 2 (ViT-L/14 + LoRA Stage 2) 결과 검증")
    print("=" * 72)

    if not Path(CKPT).exists():
        raise FileNotFoundError(f"projector.pt 없음: {CKPT}")
    if not Path(LORA).exists():
        raise FileNotFoundError(f"lora_adapter 없음: {LORA}")

    img_dog = Image.open(IMG_DOG)
    img_pika = Image.open(IMG_PIKA)
    print(f"[img-dog ] {IMG_DOG} ({img_dog.size[0]}x{img_dog.size[1]} {img_dog.mode})")
    print(f"[img-pika] {IMG_PIKA} ({img_pika.size[0]}x{img_pika.size[1]} {img_pika.mode})")

    print(f"\n[init] VLMInference 로드 중 (ViT-L/14, bf16) ...")
    print(f"  projector: {CKPT}")
    print(f"  lora     : {LORA}")
    infer = VLMInference(
        checkpoint_path=CKPT,
        lora_adapter_path=LORA,
        vision_model=VISION_MODEL_L14,
        torch_dtype=torch.bfloat16,
    )
    print(f"[init] 로드 완료\n")

    gen_cfg = GenerationConfig(
        max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True
    )

    tests = [
        ("Test A — 영문 sanity (개 이미지)", img_dog, "Describe this image briefly."),
        ("Test B — 한국어 forgetting (개 이미지)", img_dog, "이 이미지에 무엇이 보이나요?"),
        ("Test C-EN — Pikachu (영문)", img_pika, "What is in this image?"),
        ("Test C-KO — Pikachu (한국어)", img_pika, "이 이미지에 무엇이 보이나요?"),
    ]

    for label, img, q in tests:
        print("=" * 72)
        print(label)
        print("=" * 72)
        print(f"Q: {q}")
        r = infer(img, q, gen_cfg=gen_cfg)
        print(f"A: {r['answer']}")
        print(f"   (elapsed: {r['elapsed']:.2f}s)")
        print()

    print("=" * 72)
    print("v3 Step 1 vs Step 2 비교 기준")
    print("=" * 72)
    print("Step 1 (ViT-B/32) 응답 (어제 측정):")
    print("  Test A: 'In this image I can see a cat, in the background...'  ← dog→cat ❌")
    print("  Test B: '...키가 크고 바위와 레시피를 찍은 액세서리의 케이크...' ← hallucination ❌")
    print()
    print("Step 2 (ViT-L/14) 기대 효과:")
    print("  Test A: dog 으로 정확 인식 → 576 patch / ViT-L/14 vision 효과 입증")
    print("  Test B: 개 관련 한국어 응답 → vision 정확성 + forgetting 해소 둘 다")
    print("  Test C: 새 이미지로 추가 검증 (학습 데이터에 없는 캐릭터)")
    print("=" * 72)


if __name__ == "__main__":
    main()
