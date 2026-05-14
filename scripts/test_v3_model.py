"""v3 모델 검증 — Test A (영문) + Test B (한국어) + Test C (OOD/만화).

여러 v3 단계의 모델 비교 가능 (Step 1, Step 2 등).

Test 디자인:
  A — 영문 sanity      : source_dog.jpg + "Describe this image briefly."
  B — 한국어 forgetting : source_dog.jpg + "이 이미지에 무엇이 보이나요?"
  C — OOD 인식 (만화)  : source_pikachu.png + "What is in this image?"

사용:
  # Step 2 모델 (ViT-L/14, bf16)
  python scripts/test_v3_model.py \\
    --checkpoint checkpoints/v3_step2_stage2/projector.pt \\
    --lora checkpoints/v3_step2_stage2/lora_adapter \\
    --vision openai/clip-vit-large-patch14-336 --bf16

  # Step 1 모델 (ViT-B/32, fp32)
  python scripts/test_v3_model.py \\
    --checkpoint checkpoints/v3_step1_korean/projector.pt \\
    --lora checkpoints/v3_step1_korean/lora_adapter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import VISION_MODEL, GenerationConfig  # noqa: E402
from src.infer import VLMInference  # noqa: E402


TESTS = [
    ("A", "영문 sanity", "assets/source_dog.jpg",
     "Describe this image briefly."),
    ("B", "한국어 forgetting 해소", "assets/source_dog.jpg",
     "이 이미지에 무엇이 보이나요?"),
    ("C", "OOD (만화 캐릭터)", "assets/source_pikachu.png",
     "What is in this image?"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="projector.pt path")
    p.add_argument("--lora", required=True, help="lora_adapter dir")
    p.add_argument("--vision", default=VISION_MODEL,
                   help=f"vision encoder (기본 {VISION_MODEL})")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 로 inference (학습이 bf16 였으면 권장)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print("=" * 70)
    print(f"v3 모델 검증")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  lora       : {args.lora}")
    print(f"  vision     : {args.vision}")
    print(f"  dtype      : {dtype}")
    print("=" * 70)

    print(f"\n[init] VLMInference 로드 ...")
    infer = VLMInference(
        checkpoint_path=args.checkpoint,
        lora_adapter_path=args.lora,
        vision_model=args.vision,
        torch_dtype=dtype,
    )
    print(f"[init] 완료\n")

    gen_cfg = GenerationConfig(
        max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True
    )

    results = []
    for tag, desc, img_path, question in TESTS:
        print("=" * 70)
        print(f"Test {tag} — {desc}")
        print("=" * 70)
        if not Path(img_path).exists():
            print(f"  ❌ 이미지 없음: {img_path}")
            continue
        img = Image.open(img_path)
        print(f"image: {img_path} ({img.size[0]}x{img.size[1]} {img.mode})")
        print(f"Q: {question}")
        r = infer(img, question, gen_cfg=gen_cfg)
        print(f"A: {r['answer']}")
        print(f"   (elapsed: {r['elapsed']:.2f}s)")
        results.append((tag, desc, r['answer'], r['elapsed']))
        print()

    # 요약
    print("=" * 70)
    print("판정 가이드 (v3 Step 2 vs Step 1 비교)")
    print("=" * 70)
    print("  A: 'dog' 라고 말하면 ✅ vs Step 1 의 'cat' 오인 → 시각 인식 개선")
    print("  B: 한국어로 (의미 있게) 답하면 ✅ → forgetting 해소 유지")
    print("     사실적 정확도 (개 묘사) 도 보면 ✅✅")
    print("  C: '만화', 'cartoon', 'drawing', '캐릭터' 류 단어 나오면 ✅ → OOD 인식")
    print("     사실로 단언하면 ❌ (OOD 인식 부족 — 항목 3 OOD module 정당화)")
    print("=" * 70)


if __name__ == "__main__":
    main()
