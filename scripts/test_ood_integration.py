"""v3 항목 3 (OOD detection) 통합 테스트.

검증 시나리오:
  - In-Dist: source_dog.jpg (실제 개 — 학습 분포 안)
    → ood_score 낮음, is_ood=False 기대
  - OOD:     source_pikachu.png (카툰 캐릭터 — 학습 분포 밖)
    → ood_score 높음, is_ood=True 기대

동작:
  1. OODDetector 로드 (CLIP-ViT-B/32 + 57 categories)
  2. MiniLLaVA + slim adapter 로드
  3. 각 이미지: model.llm.generate(output_scores=True) → first_logits 획득
  4. detector.score(image, first_logits) → CLIP signal + entropy signal
  5. is_ood / ood_score / clip_max_sim 보고

threshold = 0.5 (default) 부터 시작, 결과 따라 조정 권장.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402
from src.ood_detection import OODDetector  # noqa: E402


CKPT = "checkpoints/v3_step1_korean/projector.pt"
LORA_SLIM = "checkpoints/v3_step1_korean/lora_adapter_slim"

TESTS = [
    ("In-Dist (실제 개)", "assets/source_dog.jpg", "What is in this image?", False),
    ("OOD (Pikachu)", "assets/source_pikachu.png", "What is in this image?", True),
]


def get_first_logits(model: MiniLLaVA, image: Image.Image, question: str) -> torch.Tensor:
    """model.generate 한 step → 첫 토큰의 logits."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"].to(model.llm.device)
    input_ids, attn = encode_for_inference(model.tokenizer, question)
    input_ids = input_ids.unsqueeze(0).to(model.llm.device)
    attn = attn.unsqueeze(0).to(model.llm.device)

    text_embeds = model.llm.get_input_embeddings()(input_ids)
    image_embeds = model.encode_image(pixel_values)
    merged_embeds, merged_mask, _ = model._merge(text_embeds, attn, image_embeds, input_ids, labels=None)

    with torch.no_grad():
        out = model.llm.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    # out.scores 는 list of tensors (각 step) — 첫 토큰의 logits
    return out.scores[0][0]  # (vocab_size,)


def main():
    print("=" * 72)
    print("v3 항목 3 (OOD detection) 통합 테스트")
    print("=" * 72)

    # 1. OOD detector
    print("\n[init] OODDetector 로드 (CLIP-ViT-B/32, 57 categories, threshold=0.5) ...")
    detector = OODDetector(threshold=0.5)
    print("[init] OODDetector 준비 완료")

    # 2. MiniLLaVA + slim adapter
    print("\n[init] MiniLLaVA + slim adapter 로드 ...")
    model = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
    model.load_projector(CKPT, map_location="cpu")
    model.load_lora_adapter(LORA_SLIM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[init] MiniLLaVA on {device} 완료\n")

    # 3. 각 테스트 실행
    results = []
    for label, img_path, q, expected_ood in TESTS:
        print("=" * 72)
        print(f"  {label}")
        print(f"  image: {img_path}")
        print("=" * 72)
        img = Image.open(img_path)

        first_logits = get_first_logits(model, img, q)
        result = detector.score(img, first_logits=first_logits)

        match_sym = "✅" if result.is_ood == expected_ood else "❌"
        print(f"  Q: {q}")
        print(f"  CLIP signal:")
        print(f"    clip_max_sim = {result.clip_max_sim:.4f}")
        print(f"    clip_match   = '{result.clip_match}'")
        print(f"  LLM signal:")
        print(f"    llm_entropy  = {result.llm_entropy:.4f} nats")
        print(f"  통합:")
        print(f"    ood_score    = {result.ood_score:.4f}")
        print(f"    is_ood       = {result.is_ood} (expected {expected_ood}) {match_sym}")
        print()
        results.append((label, expected_ood, result))

    # 4. 종합
    print("=" * 72)
    print("  종합")
    print("=" * 72)
    correct = sum(1 for _, exp, r in results if r.is_ood == exp)
    print(f"  정확도: {correct} / {len(results)}")
    for label, exp, r in results:
        sym = "✅" if r.is_ood == exp else "❌"
        print(f"    {sym} {label}: ood_score={r.ood_score:.3f} clip_sim={r.clip_max_sim:.3f} → {r.is_ood} (expected {exp})")

    # 5. 결론
    print()
    if correct == len(results):
        print("  ✅ OOD module 통합 정상 동작. 현재 threshold=0.5 적합.")
    else:
        print("  ⚠️  threshold 조정 또는 가중치 재조정 필요.")
        # 권장 threshold 추정
        scores = [(label, exp, r.ood_score) for label, exp, r in results]
        in_dist_max = max((s for l, e, s in scores if not e), default=0.0)
        ood_min = min((s for l, e, s in scores if e), default=1.0)
        if in_dist_max < ood_min:
            mid = (in_dist_max + ood_min) / 2
            print(f"  추천 threshold: {mid:.3f} (in_dist max {in_dist_max:.3f} < OOD min {ood_min:.3f})")
        else:
            print(f"  CLIP / LLM 신호로 분리 안 됨 — 다른 가중치 (weight_clip 조정) 또는 categories 확장 필요")


if __name__ == "__main__":
    main()
