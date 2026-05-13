"""다운로드된 이미지들로 KoLLaVA manifest 만 별도 생성 (recovery 용)."""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

img_dir = Path("data/korean_subset/images")
downloaded = set(p.name for p in img_dir.glob("*.jpg"))
print(f"[recover] 디스크에 있는 이미지: {len(downloaded)}")

IMAGE_TOKEN_RE = re.compile(r"\s*<image>\s*\n?|\n?\s*<image>\s*")


def clean_question(value: str) -> str:
    return IMAGE_TOKEN_RE.sub(" ", value).strip()


def korean_ratio(t: str) -> float:
    if not t:
        return 0.0
    return sum(1 for c in t if "가" <= c <= "힣") / len(t)


print("[recover] KoLLaVA streaming + 매칭...")
ds = load_dataset("tabtoyou/KoLLaVA-Instruct-150k", split="train", streaming=True)

successful = []
target = 4000
buffer = 5500

pbar = tqdm(total=buffer, desc="matching")
for sample in ds:
    pbar.update(1)
    if len(successful) >= target or pbar.n >= buffer:
        break
    image_id = (sample.get("image") or "").strip()
    if image_id not in downloaded:
        continue
    convs = sample.get("conversations") or []
    if len(convs) < 2:
        continue
    human, gpt = convs[0], convs[1]
    if human.get("from") != "human" or gpt.get("from") != "gpt":
        continue
    question = clean_question(human.get("value", ""))
    answer = (gpt.get("value") or "").strip()
    if not question or not answer or len(answer) < 4:
        continue
    img_path = (img_dir / image_id).as_posix()
    successful.append(
        {
            "image": img_path,
            "question": question,
            "answer": answer,
            "source": "kollava",
        }
    )
pbar.close()

print(f"[recover] 매칭된 valid samples: {len(successful)}")

random.Random(42).shuffle(successful)
successful = successful[:target]

manifest_path = Path("data/korean_subset/manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(successful, f, ensure_ascii=False, indent=2)

avg_q = sum(len(e["question"]) for e in successful) / len(successful)
avg_a = sum(len(e["answer"]) for e in successful) / len(successful)
avg_kor_q = sum(korean_ratio(e["question"]) for e in successful) / len(successful)
avg_kor_a = sum(korean_ratio(e["answer"]) for e in successful) / len(successful)

print()
print(f"[done] manifest: {manifest_path} ({len(successful)} samples)")
print(f"  평균 길이 — Q: {avg_q:.1f}자, A: {avg_a:.1f}자")
print(f"  Korean (Hangul) 비율 — Q: {100*avg_kor_q:.1f}%, A: {100*avg_kor_a:.1f}%")
