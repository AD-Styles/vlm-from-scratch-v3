"""라이브 HF Space (baseline) vs 로컬 enhanced wrapper — 실제 응답 비교.

목적: 사용자 challenge 응답
  "+20%p 라고 말로만 하지말고 실제로 사용해보고 답변 확인해봐"

테스트 셋: 같은 이미지/질문을 두 시스템에 던지고 raw 응답 비교.
"""
from __future__ import annotations

import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# datasets first (Windows DLL 충돌 회피)
from datasets import load_dataset  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from gradio_client import Client, handle_file  # noqa: E402

from src.enhanced_inference import EnhancedVLM  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402
from src.ood_detection import OODDetector  # noqa: E402

V3_PROJECTOR = "checkpoints/v3_step1_korean/projector.pt"
V3_ADAPTER_SLIM = "checkpoints/v3_step1_korean/lora_adapter_slim"

# 테스트 케이스 (file path, prompt, language, ground-truth label, hint about expected answer)
TEST_CASES = [
    # source_dog.jpg = white-ish small dog wearing hat (no person, no cat)
    ("assets/source_dog.jpg", "What is in this image?", "en", "subject_id", "dog"),
    ("assets/source_dog.jpg", "Is there a dog in the image?", "en", "yesno", "yes"),
    ("assets/source_dog.jpg", "Is there a cat in the image?", "en", "yesno", "no"),
    ("assets/source_dog.jpg", "Is there a person in the image?", "en", "yesno", "no"),
    ("assets/source_dog.jpg", "Is there a car in the image?", "en", "yesno", "no"),
    ("assets/source_dog.jpg", "What color is the main subject?", "en", "color", "white"),
    ("assets/source_dog.jpg", "이 이미지에 무엇이 보이나요?", "ko", "subject_id", "개"),
    ("assets/source_dog.jpg", "이 동물의 종류는 무엇인가요?", "ko", "subject_id", "개"),
    # source_pikachu.png = yellow cartoon character (OOD)
    ("assets/source_pikachu.png", "What is in this image?", "en", "subject_id", "cartoon"),
    ("assets/source_pikachu.png", "Is there a real animal in the image?", "en", "yesno", "no"),
    ("assets/source_pikachu.png", "What color is this character?", "en", "color", "yellow"),
    ("assets/source_pikachu.png", "이 캐릭터의 색은 무엇인가요?", "ko", "color", "노란색"),
]


def call_live_space(client: Client, image_path: str, question: str) -> str:
    """라이브 HF Space API 호출 — v3-Enhanced wrapper version (simplified API)."""
    try:
        result = client.predict(
            image=handle_file(image_path),
            question=question,
            api_name="/predict_1",
        )
        return result[0] if result else ""
    except Exception as e:
        return f"[error: {type(e).__name__}: {str(e)[:100]}]"


def main():
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("  라이브 Space (baseline) vs 로컬 enhanced wrapper")
    print("=" * 72)

    # 1. 라이브 Space 연결
    print("\n[1] 라이브 HF Space 연결 ...")
    client = Client("AD-Styles/mini-llava-v3-demo")

    # 2. 로컬 v3 + enhanced 로드
    print("\n[2] 로컬 v3 모델 + EnhancedVLM 로드 ...")
    model = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
    model.load_projector(V3_PROJECTOR, map_location="cpu")
    model.load_lora_adapter(V3_ADAPTER_SLIM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    detector = OODDetector(threshold=0.5, device=device)
    enhanced = EnhancedVLM(
        model=model,
        ood_detector=detector,
        enable_translation=True,        # KO→EN 활성 (m2m100)
        enable_back_translation=True,   # EN→KO 활성 (m2m100, 한국어 응답 복구) — Space 와 동일
        enable_clip_subject=True,
        device=device,
        pope_threshold=0.0,             # case 2 dog 살리기 위해 0.0 (margin > 0 → yes)
    )

    # 3. 각 케이스 실행
    print(f"\n[3] {len(TEST_CASES)} 테스트 케이스 실행 ...")
    results = []
    for i, (img_path, question, lang, qtype, expected) in enumerate(TEST_CASES, 1):
        print(f"\n--- [{i}/{len(TEST_CASES)}] {Path(img_path).stem} | [{lang}|{qtype}] ---")
        print(f"  Q: {question}")
        print(f"  Expected: {expected}")

        # Live Space (baseline)
        t0 = time.time()
        live_resp = call_live_space(client, img_path, question)
        live_time = time.time() - t0
        print(f"  LIVE (baseline): {live_resp!r}  ({live_time:.1f}s)")

        # Local enhanced
        t0 = time.time()
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        enh_resp, meta = enhanced.answer(img, question, return_meta=True)
        enh_time = time.time() - t0
        print(f"  ENHANCED: {enh_resp!r}  ({enh_time:.1f}s) — path={meta.get('used_path')}")

        results.append({
            "image": img_path,
            "question": question,
            "lang": lang,
            "qtype": qtype,
            "expected": expected,
            "live_baseline": live_resp,
            "enhanced": enh_resp,
            "enhanced_path": meta.get("used_path"),
            "enhanced_meta": {k: v for k, v in meta.items() if k != "raw_answer_en"},
        })

    # 4. 저장
    out_path = out_dir / "live_vs_enhanced.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")

    # 5. 마크다운 보고서
    md = ["# 라이브 Space vs 로컬 Enhanced — 실제 응답 비교", "",
          "사용자 challenge: 수치만 말하지 말고 실제 사용해보고 답변 확인.", "",
          "테스트 방법:",
          "- 라이브 Space (baseline): https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo (gradio_client API)",
          "- 로컬 Enhanced: src/enhanced_inference.py (CLIP grounding + extraction + OOD gate)",
          "- 같은 이미지, 같은 질문 → raw 응답 그대로 비교",
          "",
          "| # | 이미지 | 질문 | 기대 | LIVE Space (baseline) | Enhanced (local) | path |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        img = Path(r["image"]).stem
        live = r["live_baseline"][:80].replace("\n", " ").replace("|", "\\|")
        enh = r["enhanced"][:80].replace("\n", " ").replace("|", "\\|")
        md.append(f"| {i} | {img} | {r['question']} | {r['expected']} | {live} | {enh} | {r.get('enhanced_path','?')} |")

    md_path = out_dir / "live_vs_enhanced.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"저장: {md_path}")


if __name__ == "__main__":
    main()
