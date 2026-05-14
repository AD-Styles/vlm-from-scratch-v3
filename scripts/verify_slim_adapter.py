"""slim adapter 의 추론 품질 = 원본 adapter 의 추론 품질 검증.

방법:
  1. 동일 모델 (Step 1) + FULL adapter 로 N개 prompt → 응답 A 저장
  2. 동일 모델 (Step 1) + SLIM adapter 로 동일 prompt → 응답 B
  3. 모든 응답 A == B 인지 비교 (greedy decoding 으로 deterministic 보장)

판정:
  ✅ 모든 응답 정확 일치 → slim adapter 무손실 입증
  ⚠️  일부 다름 → 어떤 prompt 에서 다른지 분석 → 가설 재점검

사용:
  python scripts/verify_slim_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import GenerationConfig  # noqa: E402
from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402


CKPT = "checkpoints/v3_step1_korean/projector.pt"
LORA_FULL = "checkpoints/v3_step1_korean/lora_adapter"
LORA_SLIM = "checkpoints/v3_step1_korean/lora_adapter_slim"
IMG_DOG = "assets/source_dog.jpg"
IMG_PIKA = "assets/source_pikachu.png"


# 검증용 prompt (다양한 길이 + 한국어/영문 + 다른 이미지)
TEST_PROMPTS = [
    (IMG_DOG, "Describe this image briefly."),
    (IMG_DOG, "What animal is in this image?"),
    (IMG_DOG, "이 이미지에 무엇이 보이나요?"),
    (IMG_DOG, "이 이미지의 색상은 무엇인가요?"),
    (IMG_PIKA, "What is in this image?"),
    (IMG_PIKA, "Describe this character."),
    (IMG_PIKA, "이 캐릭터의 색은?"),
]


def generate_greedy(model: MiniLLaVA, image: Image.Image, question: str, max_new_tokens: int = 64) -> str:
    """greedy decoding (do_sample=False) 으로 deterministic output."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"].to(model.llm.device)
    input_ids, attention_mask = encode_for_inference(model.tokenizer, question)
    input_ids = input_ids.unsqueeze(0).to(model.llm.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.llm.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # ★ deterministic
            temperature=1.0,
            top_p=1.0,
        )
    return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def run_with_adapter(adapter_path: str, label: str) -> dict:
    print()
    print("=" * 72)
    print(f"  {label}: {adapter_path}")
    print("=" * 72)

    model = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
    model.load_projector(CKPT, map_location="cpu")
    model.load_lora_adapter(adapter_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = {}
    img_cache = {}
    for img_path, q in TEST_PROMPTS:
        if img_path not in img_cache:
            img_cache[img_path] = Image.open(img_path)
        img = img_cache[img_path]
        ans = generate_greedy(model, img, q)
        key = f"{Path(img_path).stem}__{q[:40]}"
        results[key] = ans
        print(f"  Q[{Path(img_path).stem}]: {q}")
        print(f"  A: {ans}")
        print()

    # GPU 메모리 확보
    del model
    torch.cuda.empty_cache()

    return results


def main():
    if not Path(LORA_SLIM).exists():
        raise FileNotFoundError(
            f"slim adapter 없음: {LORA_SLIM}\n"
            f"먼저 실행: python scripts/extract_lora_v3.py "
            f"--input-dir {LORA_FULL} --output-dir {LORA_SLIM}"
        )

    print("[step 1/2] FULL adapter (1045MB) 로 추론")
    full_results = run_with_adapter(LORA_FULL, "FULL")

    print()
    print("[step 2/2] SLIM adapter (~8MB) 로 추론")
    slim_results = run_with_adapter(LORA_SLIM, "SLIM")

    # ────────── 비교 ──────────
    print()
    print("=" * 72)
    print("  결과 비교 — 모든 prompt 의 응답 일치 여부")
    print("=" * 72)
    matches = 0
    total = len(TEST_PROMPTS)
    for key in full_results:
        full_ans = full_results[key]
        slim_ans = slim_results.get(key, "<missing>")
        if full_ans == slim_ans:
            matches += 1
            print(f"  ✅ {key[:60]:<60} | match")
        else:
            print(f"  ❌ {key[:60]:<60} | DIFFER")
            print(f"     FULL: {full_ans[:100]}")
            print(f"     SLIM: {slim_ans[:100]}")

    print()
    print("=" * 72)
    print(f"  최종: {matches} / {total} 일치 ({100 * matches / total:.0f}%)")
    print("=" * 72)
    if matches == total:
        print("  ✅✅✅ slim adapter 무손실 입증 — 1GB → 8MB 안전 deploy 가능")
    elif matches >= total * 0.9:
        print("  ⚠️  대부분 일치하지만 일부 다름 — 분석 필요")
    else:
        print("  🔴 slim adapter 가 품질 손실 — slim 화 가설 재검토 필요")


if __name__ == "__main__":
    main()
