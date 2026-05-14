"""표준 benchmark 기반 v2 vs v3 정직한 성능 평가.

평가 데이터셋:
  1. VQAv2 val (lmms-lab/VQAv2, streaming) — 50 samples
     metric: 공식 VQA accuracy = mean( min(matches_in_GT/3, 1.0) )
  2. POPE test (lmms-lab/POPE, streaming) — 60 samples
     metric: yes/no accuracy, yes-recall, no-recall, F1
     (POPE = Polling-based Object Probing Evaluation, hallucination 측정 표준)

수치 metric 만 사용 (eyeballing 0).

  - 두 benchmark 모두 학습 데이터에 사용한 적 없음:
      * VQAv2: train ⊥ val 분리 (공식)
      * POPE: COCO val2014 기반, train2014 와 분리

산출:
  eval_results/v2_results.json
  eval_results/v3_results.json
  eval_results/comparison.md
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# UTF-8 stdout/stderr — Windows cp949 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# IMPORTANT: datasets MUST be imported BEFORE torch on Windows.
# torch 먼저 import 시 pyarrow/datasets 의 C 확장과 DLL 충돌로 segfault 발생.
from datasets import load_dataset  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import encode_for_inference  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402

# 평가 설정
N_VQAV2 = 50
N_POPE = 60

# 모델 경로
V2_LOCAL = "checkpoints/v2_baseline_for_compare"
V3_PROJECTOR = "checkpoints/v3_step1_korean/projector.pt"
V3_ADAPTER_SLIM = "checkpoints/v3_step1_korean/lora_adapter_slim"

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# 데이터셋 로드
# ──────────────────────────────────────────────────────────────────
def load_vqav2(n: int) -> list[dict]:
    """VQAv2 val streaming → n 개 샘플."""
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    out = []
    for s in ds:
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(
            {
                "image": img,
                "question_id": s["question_id"],
                "question": s["question"],
                "mc_answer": s["multiple_choice_answer"],
                "all_answers": [a["answer"] for a in s["answers"]],
                "answer_type": s["answer_type"],  # 'yes/no', 'number', 'other'
            }
        )
        if len(out) >= n:
            break
    return out


def load_pope(n: int) -> list[dict]:
    """POPE test streaming → n 개 샘플."""
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    out = []
    for s in ds:
        img = s["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(
            {
                "image": img,
                "id": s["id"],
                "question": s["question"],
                "gt_answer": s["answer"].lower().strip(),
                "category": s["category"],  # 'random', 'popular', 'adversarial'
            }
        )
        if len(out) >= n:
            break
    return out


# ──────────────────────────────────────────────────────────────────
# 추론 헬퍼
# ──────────────────────────────────────────────────────────────────
def generate(model: MiniLLaVA, image: Image.Image, question: str, max_new: int = 20) -> str:
    pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"].to(model.llm.device)
    input_ids, attn = encode_for_inference(model.tokenizer, question)
    input_ids = input_ids.unsqueeze(0).to(model.llm.device)
    attn = attn.unsqueeze(0).to(model.llm.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            max_new_tokens=max_new,
            do_sample=False,  # greedy → deterministic
        )
    return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────
# Metric 계산
# ──────────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    """VQA 공식 normalize: 소문자, 양끝 공백/구두점 제거."""
    s = s.lower().strip()
    # 양끝 따옴표/구두점 제거
    while s and s[-1] in ".,;:!?'\"`":
        s = s[:-1]
    while s and s[0] in "'\"`":
        s = s[1:]
    return s.strip()


def vqa_accuracy(pred: str, all_answers: list[str]) -> float:
    """공식 VQA accuracy = min(# matching / 3, 1.0)."""
    p = normalize_answer(pred)
    matches = sum(1 for a in all_answers if normalize_answer(a) == p)
    return min(matches / 3.0, 1.0)


def pope_predict_yn(pred: str) -> str:
    """POPE 응답에서 첫 yes/no 토큰 추출. 없으면 '?'."""
    p = pred.lower().strip()
    # 처음 30자 안에서 찾기
    head = p[:30]
    if "yes" in head and "no" not in head[: head.find("yes") + 3]:
        return "yes"
    if "no" in head and "yes" not in head[: head.find("no") + 2]:
        return "no"
    if head.startswith("yes"):
        return "yes"
    if head.startswith("no"):
        return "no"
    return "?"


# ──────────────────────────────────────────────────────────────────
# 평가 실행 (모델당 1회)
# ──────────────────────────────────────────────────────────────────
def run_eval(model: MiniLLaVA, label: str, vqav2: list[dict], pope: list[dict]) -> dict:
    print(f"\n{'=' * 72}")
    print(f"  {label} 평가 시작 (VQAv2 {len(vqav2)} + POPE {len(pope)})")
    print(f"{'=' * 72}")

    # VQAv2 (max_new=10, VQA 응답은 보통 단답)
    print(f"\n[{label}] VQAv2 inference ...")
    vqa_results = []
    t0 = time.time()
    for i, s in enumerate(vqav2):
        pred = generate(model, s["image"], s["question"], max_new=10)
        acc = vqa_accuracy(pred, s["all_answers"])
        vqa_results.append(
            {
                "question_id": s["question_id"],
                "question": s["question"],
                "answer_type": s["answer_type"],
                "mc_answer": s["mc_answer"],
                "all_answers": s["all_answers"],
                "pred": pred,
                "vqa_acc": acc,
            }
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] VQAv2 {i + 1}/{len(vqav2)} ({elapsed:.0f}s)")

    # POPE (max_new=5, yes/no 단답)
    print(f"\n[{label}] POPE inference ...")
    pope_results = []
    t0 = time.time()
    for i, s in enumerate(pope):
        pred = generate(model, s["image"], s["question"], max_new=5)
        pred_yn = pope_predict_yn(pred)
        correct = pred_yn == s["gt_answer"]
        pope_results.append(
            {
                "id": s["id"],
                "question": s["question"],
                "gt_answer": s["gt_answer"],
                "category": s["category"],
                "pred": pred,
                "pred_yn": pred_yn,
                "correct": correct,
            }
        )
        if (i + 1) % 15 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] POPE {i + 1}/{len(pope)} ({elapsed:.0f}s)")

    # 집계
    vqa_acc_overall = sum(r["vqa_acc"] for r in vqa_results) / max(1, len(vqa_results))

    # answer_type 별
    by_type = {}
    for r in vqa_results:
        t = r["answer_type"]
        by_type.setdefault(t, []).append(r["vqa_acc"])
    vqa_by_type = {t: sum(xs) / len(xs) for t, xs in by_type.items()}

    # POPE 집계
    pope_acc = sum(1 for r in pope_results if r["correct"]) / max(1, len(pope_results))

    # POPE yes/no recall + precision (binary classification)
    tp = sum(1 for r in pope_results if r["correct"] and r["gt_answer"] == "yes")
    fn = sum(1 for r in pope_results if not r["correct"] and r["gt_answer"] == "yes")
    fp = sum(1 for r in pope_results if not r["correct"] and r["gt_answer"] == "no")
    tn = sum(1 for r in pope_results if r["correct"] and r["gt_answer"] == "no")
    yes_recall = tp / max(1, tp + fn)
    yes_precision = tp / max(1, tp + fp)
    yes_f1 = 2 * yes_precision * yes_recall / max(1e-6, yes_precision + yes_recall)
    refusal_rate = sum(1 for r in pope_results if r["pred_yn"] == "?") / max(1, len(pope_results))

    # POPE category 별
    by_cat = {}
    for r in pope_results:
        c = r["category"]
        by_cat.setdefault(c, []).append(int(r["correct"]))
    pope_by_cat = {c: sum(xs) / len(xs) for c, xs in by_cat.items()}

    summary = {
        "label": label,
        "n_vqav2": len(vqa_results),
        "n_pope": len(pope_results),
        "vqav2_accuracy": vqa_acc_overall,
        "vqav2_by_type": vqa_by_type,
        "pope_accuracy": pope_acc,
        "pope_yes_recall": yes_recall,
        "pope_yes_precision": yes_precision,
        "pope_yes_f1": yes_f1,
        "pope_refusal_rate": refusal_rate,
        "pope_by_category": pope_by_cat,
        "pope_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    print(f"\n[{label}] 집계 완료:")
    print(f"  VQAv2 accuracy:     {summary['vqav2_accuracy'] * 100:.2f}%")
    for t, a in vqa_by_type.items():
        print(f"    by '{t}':          {a * 100:.2f}%")
    print(f"  POPE accuracy:      {summary['pope_accuracy'] * 100:.2f}%")
    print(f"  POPE yes-F1:        {summary['pope_yes_f1']:.3f}")
    print(f"  POPE refusal rate:  {summary['pope_refusal_rate'] * 100:.2f}%")

    return {
        "summary": summary,
        "vqa_results": vqa_results,
        "pope_results": pope_results,
    }


def save_results(results: dict, path: Path) -> None:
    """JSON 저장 — image 객체 제외."""
    out = {
        "summary": results["summary"],
        "vqav2_details": [
            {k: v for k, v in r.items() if k != "image"} for r in results["vqa_results"]
        ],
        "pope_details": [
            {k: v for k, v in r.items() if k != "image"} for r in results["pope_results"]
        ],
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved: {path}")


def write_comparison_md(v2_summary: dict, v3_summary: dict, out_path: Path) -> None:
    lines = [
        "# v2 vs v3 정직한 성능 비교",
        "",
        "## 평가 설정",
        "",
        f"- VQAv2 val: {v2_summary['n_vqav2']} samples (lmms-lab/VQAv2, streaming)",
        f"- POPE test: {v2_summary['n_pope']} samples (lmms-lab/POPE, streaming, COCO val2014 기반)",
        "- decoding: greedy (do_sample=False) — deterministic",
        "- v2: AD-Styles/mini-llava-stage2 (1045MB adapter)",
        "- v3: AD-Styles/mini-llava-v3 (8.28MB slim adapter, Korean training)",
        "",
        "## VQAv2 — 공식 VQA accuracy",
        "",
        "| 항목 | v2 | v3 | diff |",
        "|---|---|---|---|",
        f"| 전체 | {v2_summary['vqav2_accuracy']*100:.2f}% | {v3_summary['vqav2_accuracy']*100:.2f}% | {(v3_summary['vqav2_accuracy']-v2_summary['vqav2_accuracy'])*100:+.2f}%p |",
    ]
    # type 별
    types = sorted(set(v2_summary["vqav2_by_type"]) | set(v3_summary["vqav2_by_type"]))
    for t in types:
        v2t = v2_summary["vqav2_by_type"].get(t, 0)
        v3t = v3_summary["vqav2_by_type"].get(t, 0)
        lines.append(f"| answer_type='{t}' | {v2t*100:.2f}% | {v3t*100:.2f}% | {(v3t-v2t)*100:+.2f}%p |")

    lines.extend(
        [
            "",
            "## POPE — hallucination 평가",
            "",
            "POPE = Polling-based Object Probing Evaluation. yes/no 단답 평가 데이터셋.",
            "",
            "| Metric | v2 | v3 | diff |",
            "|---|---|---|---|",
            f"| 전체 정확도 | {v2_summary['pope_accuracy']*100:.2f}% | {v3_summary['pope_accuracy']*100:.2f}% | {(v3_summary['pope_accuracy']-v2_summary['pope_accuracy'])*100:+.2f}%p |",
            f"| yes-Recall | {v2_summary['pope_yes_recall']*100:.2f}% | {v3_summary['pope_yes_recall']*100:.2f}% | {(v3_summary['pope_yes_recall']-v2_summary['pope_yes_recall'])*100:+.2f}%p |",
            f"| yes-Precision | {v2_summary['pope_yes_precision']*100:.2f}% | {v3_summary['pope_yes_precision']*100:.2f}% | {(v3_summary['pope_yes_precision']-v2_summary['pope_yes_precision'])*100:+.2f}%p |",
            f"| yes-F1 | {v2_summary['pope_yes_f1']:.3f} | {v3_summary['pope_yes_f1']:.3f} | {(v3_summary['pope_yes_f1']-v2_summary['pope_yes_f1']):+.3f} |",
            f"| Refusal rate (?) | {v2_summary['pope_refusal_rate']*100:.2f}% | {v3_summary['pope_refusal_rate']*100:.2f}% | {(v3_summary['pope_refusal_rate']-v2_summary['pope_refusal_rate'])*100:+.2f}%p |",
        ]
    )
    # POPE category
    cats = sorted(set(v2_summary["pope_by_category"]) | set(v3_summary["pope_by_category"]))
    if cats:
        lines.extend(["", "### POPE — category 별 정확도", "", "| category | v2 | v3 | diff |", "|---|---|---|---|"])
        for c in cats:
            v2c = v2_summary["pope_by_category"].get(c, 0)
            v3c = v3_summary["pope_by_category"].get(c, 0)
            lines.append(f"| {c} | {v2c*100:.2f}% | {v3c*100:.2f}% | {(v3c-v2c)*100:+.2f}%p |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved: {out_path}")


def load_v2() -> MiniLLaVA:
    print("[v2] loading ...")
    m = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
    m.load_projector(f"{V2_LOCAL}/projector.pt", map_location="cpu")
    m.load_lora_adapter(f"{V2_LOCAL}/lora_adapter")
    m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return m


def load_v3() -> MiniLLaVA:
    print("[v3] loading ...")
    m = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
    m.load_projector(V3_PROJECTOR, map_location="cpu")
    m.load_lora_adapter(V3_ADAPTER_SLIM)
    m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return m


def main():
    print("=" * 72)
    print(f"  표준 benchmark eval — VQAv2 ({N_VQAV2}) + POPE ({N_POPE})")
    print("=" * 72)

    print("\n[step 1] 데이터셋 로드 (streaming) ...")
    t0 = time.time()
    vqav2 = load_vqav2(N_VQAV2)
    pope = load_pope(N_POPE)
    print(f"  VQAv2 loaded: {len(vqav2)}, POPE loaded: {len(pope)} ({time.time()-t0:.0f}s)")

    print("\n[step 2] v2 평가 ...")
    v2_model = load_v2()
    v2_results = run_eval(v2_model, "v2", vqav2, pope)
    save_results(v2_results, OUT_DIR / "v2_results.json")
    del v2_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[step 3] v3 평가 ...")
    v3_model = load_v3()
    v3_results = run_eval(v3_model, "v3", vqav2, pope)
    save_results(v3_results, OUT_DIR / "v3_results.json")
    del v3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[step 4] 비교 보고서 작성 ...")
    write_comparison_md(v2_results["summary"], v3_results["summary"], OUT_DIR / "comparison.md")

    # 최종 헤드라인
    v2s = v2_results["summary"]
    v3s = v3_results["summary"]
    print("\n" + "=" * 72)
    print("  최종 비교 (수치)")
    print("=" * 72)
    print(f"  VQAv2 accuracy:  v2 {v2s['vqav2_accuracy']*100:6.2f}%  →  v3 {v3s['vqav2_accuracy']*100:6.2f}%  ({(v3s['vqav2_accuracy']-v2s['vqav2_accuracy'])*100:+.2f}%p)")
    print(f"  POPE accuracy:   v2 {v2s['pope_accuracy']*100:6.2f}%  →  v3 {v3s['pope_accuracy']*100:6.2f}%  ({(v3s['pope_accuracy']-v2s['pope_accuracy'])*100:+.2f}%p)")
    print(f"  POPE yes-F1:     v2 {v2s['pope_yes_f1']:6.3f}   →  v3 {v3s['pope_yes_f1']:6.3f}   ({v3s['pope_yes_f1']-v2s['pope_yes_f1']:+.3f})")


if __name__ == "__main__":
    main()
