"""V3-Enhanced wrapper 평가 — baseline 과 동일한 VQAv2/POPE 셋에서 측정.

비교:
  - v3-baseline (raw v3 model.generate, eval_proper.py 결과)
  - v3-enhanced (EnhancedVLM wrapper: CLIP grounding + extraction + OOD gate + translation)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# datasets first (Windows DLL 충돌 회피)
from datasets import load_dataset  # noqa: E402
import torch  # noqa: E402

from src.dataset import encode_for_inference  # noqa: E402
from src.enhanced_inference import EnhancedVLM  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402
from src.ood_detection import OODDetector  # noqa: E402

# eval_proper.py 와 동일 설정
N_VQAV2 = 50
N_POPE = 60

V3_PROJECTOR = "checkpoints/v3_step1_korean/projector.pt"
V3_ADAPTER_SLIM = "checkpoints/v3_step1_korean/lora_adapter_slim"

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)


def load_vqav2(n: int) -> list[dict]:
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
                "answer_type": s["answer_type"],
            }
        )
        if len(out) >= n:
            break
    return out


def load_pope(n: int) -> list[dict]:
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
                "category": s["category"],
            }
        )
        if len(out) >= n:
            break
    return out


def normalize(s: str) -> str:
    s = s.lower().strip()
    while s and s[-1] in ".,;:!?'\"`":
        s = s[:-1]
    while s and s[0] in "'\"`":
        s = s[1:]
    return s.strip()


def vqa_accuracy(pred: str, all_answers: list[str]) -> float:
    p = normalize(pred)
    matches = sum(1 for a in all_answers if normalize(a) == p)
    return min(matches / 3.0, 1.0)


def pope_predict_yn(pred: str) -> str:
    p = pred.lower().strip()
    head = p[:30]
    yes_pos = head.find("yes")
    no_pos = head.find("no")
    if yes_pos == -1 and no_pos == -1:
        return "?"
    if yes_pos == -1:
        return "no"
    if no_pos == -1:
        return "yes"
    return "yes" if yes_pos < no_pos else "no"


def main():
    print("=" * 72)
    print("  V3-Enhanced eval (CLIP grounding + extraction + OOD gate + Korean MT)")
    print("=" * 72)

    print("\n[1] 로딩: 데이터셋 ...")
    t0 = time.time()
    vqav2 = load_vqav2(N_VQAV2)
    pope = load_pope(N_POPE)
    print(f"  VQAv2 {len(vqav2)}, POPE {len(pope)} ({time.time()-t0:.0f}s)")

    print("\n[2] 로딩: v3 model + OOD detector + EnhancedVLM ...")
    model = MiniLLaVA(freeze_vision=True, freeze_llm=True, torch_dtype=torch.float32)
    model.load_projector(V3_PROJECTOR, map_location="cpu")
    model.load_lora_adapter(V3_ADAPTER_SLIM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    detector = OODDetector(threshold=0.5, device=device)

    enhanced = EnhancedVLM(
        model=model,
        ood_detector=detector,
        enable_translation=False,  # 영문 eval 만 하므로 MT 불필요 (시간/메모리 절약)
        enable_clip_subject=True,
        device=device,
    )

    # ──────────────────────────────────────────────────────────────────
    # VQAv2
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[3] V3-Enhanced VQAv2 inference ({N_VQAV2}) ...")
    vqa_results = []
    paths = {}
    t0 = time.time()
    for i, s in enumerate(vqav2):
        pred, meta = enhanced.answer(s["image"], s["question"], return_meta=True)
        acc = vqa_accuracy(pred, s["all_answers"])
        used = meta.get("used_path", "unknown")
        paths[used] = paths.get(used, 0) + 1
        vqa_results.append(
            {
                "question_id": s["question_id"],
                "question": s["question"],
                "answer_type": s["answer_type"],
                "mc_answer": s["mc_answer"],
                "all_answers": s["all_answers"],
                "pred": pred,
                "vqa_acc": acc,
                "used_path": used,
                "raw_answer_en": meta.get("raw_answer_en", ""),
                "clip_override": meta.get("clip_override", False),
                "clip_color": meta.get("clip_color"),
                "clip_subject_label": meta.get("clip_subject_label"),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  VQAv2 {i + 1}/{len(vqav2)} ({time.time()-t0:.0f}s) — paths so far: {paths}")
    vqa_acc_overall = sum(r["vqa_acc"] for r in vqa_results) / max(1, len(vqa_results))

    # ──────────────────────────────────────────────────────────────────
    # POPE
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[4] V3-Enhanced POPE inference ({N_POPE}) ...")
    pope_results = []
    t0 = time.time()
    for i, s in enumerate(pope):
        pred, meta = enhanced.answer(s["image"], s["question"], return_meta=True)
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
                "used_path": meta.get("used_path"),
                "clip_grounding_obj": meta.get("clip_grounding_obj"),
                "clip_grounding_margin": meta.get("clip_grounding_margin"),
            }
        )
        if (i + 1) % 15 == 0:
            print(f"  POPE {i + 1}/{len(pope)} ({time.time()-t0:.0f}s)")

    # ──────────────────────────────────────────────────────────────────
    # 집계
    # ──────────────────────────────────────────────────────────────────
    pope_acc = sum(1 for r in pope_results if r["correct"]) / max(1, len(pope_results))
    tp = sum(1 for r in pope_results if r["correct"] and r["gt_answer"] == "yes")
    fn = sum(1 for r in pope_results if not r["correct"] and r["gt_answer"] == "yes")
    fp = sum(1 for r in pope_results if not r["correct"] and r["gt_answer"] == "no")
    tn = sum(1 for r in pope_results if r["correct"] and r["gt_answer"] == "no")
    yes_recall = tp / max(1, tp + fn)
    yes_precision = tp / max(1, tp + fp)
    yes_f1 = 2 * yes_precision * yes_recall / max(1e-6, yes_precision + yes_recall)

    by_type = {}
    for r in vqa_results:
        by_type.setdefault(r["answer_type"], []).append(r["vqa_acc"])
    vqa_by_type = {t: sum(xs) / len(xs) for t, xs in by_type.items()}
    by_cat = {}
    for r in pope_results:
        by_cat.setdefault(r["category"], []).append(int(r["correct"]))
    pope_by_cat = {c: sum(xs) / len(xs) for c, xs in by_cat.items()}

    summary = {
        "label": "v3-enhanced",
        "n_vqav2": len(vqa_results),
        "n_pope": len(pope_results),
        "vqav2_accuracy": vqa_acc_overall,
        "vqav2_by_type": vqa_by_type,
        "pope_accuracy": pope_acc,
        "pope_yes_recall": yes_recall,
        "pope_yes_precision": yes_precision,
        "pope_yes_f1": yes_f1,
        "pope_by_category": pope_by_cat,
        "pope_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "vqa_paths": paths,
    }

    out = {
        "summary": summary,
        "vqav2_details": vqa_results,
        "pope_details": pope_results,
    }
    save_path = OUT_DIR / "v3_enhanced_results.json"
    save_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  saved: {save_path}")

    # ──────────────────────────────────────────────────────────────────
    # 비교 표 (vs baseline)
    # ──────────────────────────────────────────────────────────────────
    v2 = json.load(open(OUT_DIR / "v2_results.json", encoding="utf-8"))["summary"]
    v3b = json.load(open(OUT_DIR / "v3_results.json", encoding="utf-8"))["summary"]
    v3e = summary

    md = [
        "# 최종 성능 비교 — v2 vs v3-baseline vs v3-enhanced",
        "",
        f"평가: VQAv2 val {N_VQAV2} + POPE test {N_POPE}, greedy decoding.",
        "",
        "## VQAv2 (공식 VQA accuracy)",
        "",
        "| 항목 | v2 | v3-baseline | v3-enhanced | enhanced - baseline |",
        "|---|---|---|---|---|",
        f"| 전체 | {v2['vqav2_accuracy']*100:.2f}% | {v3b['vqav2_accuracy']*100:.2f}% | **{v3e['vqav2_accuracy']*100:.2f}%** | {(v3e['vqav2_accuracy']-v3b['vqav2_accuracy'])*100:+.2f}%p |",
    ]
    types = sorted(set(v2["vqav2_by_type"]) | set(v3b["vqav2_by_type"]) | set(v3e["vqav2_by_type"]))
    for t in types:
        md.append(
            f"| answer_type='{t}' | "
            f"{v2['vqav2_by_type'].get(t, 0)*100:.2f}% | "
            f"{v3b['vqav2_by_type'].get(t, 0)*100:.2f}% | "
            f"**{v3e['vqav2_by_type'].get(t, 0)*100:.2f}%** | "
            f"{(v3e['vqav2_by_type'].get(t, 0)-v3b['vqav2_by_type'].get(t, 0))*100:+.2f}%p |"
        )
    md.extend(
        [
            "",
            "## POPE (hallucination 평가)",
            "",
            "| Metric | v2 | v3-baseline | v3-enhanced | enhanced - baseline |",
            "|---|---|---|---|---|",
            f"| 전체 정확도 | {v2['pope_accuracy']*100:.2f}% | {v3b['pope_accuracy']*100:.2f}% | **{v3e['pope_accuracy']*100:.2f}%** | {(v3e['pope_accuracy']-v3b['pope_accuracy'])*100:+.2f}%p |",
            f"| yes-Recall | {v2['pope_yes_recall']*100:.2f}% | {v3b['pope_yes_recall']*100:.2f}% | **{v3e['pope_yes_recall']*100:.2f}%** | {(v3e['pope_yes_recall']-v3b['pope_yes_recall'])*100:+.2f}%p |",
            f"| yes-Precision | {v2['pope_yes_precision']*100:.2f}% | {v3b['pope_yes_precision']*100:.2f}% | **{v3e['pope_yes_precision']*100:.2f}%** | {(v3e['pope_yes_precision']-v3b['pope_yes_precision'])*100:+.2f}%p |",
            f"| yes-F1 | {v2['pope_yes_f1']:.3f} | {v3b['pope_yes_f1']:.3f} | **{v3e['pope_yes_f1']:.3f}** | {v3e['pope_yes_f1']-v3b['pope_yes_f1']:+.3f} |",
            "",
            f"## V3-Enhanced 라우팅 통계 (VQAv2 {N_VQAV2} 샘플)",
            "",
            "| Path | 사용 횟수 |",
            "|---|---|",
        ]
    )
    for p, c in sorted(paths.items(), key=lambda x: -x[1]):
        md.append(f"| {p} | {c} |")

    cmp_path = OUT_DIR / "comparison_enhanced.md"
    cmp_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  saved: {cmp_path}")

    # 최종 헤드라인
    print("\n" + "=" * 72)
    print("  최종 비교 (수치)")
    print("=" * 72)
    print(f"  VQAv2:    v2 {v2['vqav2_accuracy']*100:6.2f}%  →  v3-base {v3b['vqav2_accuracy']*100:6.2f}%  →  v3-ENH {v3e['vqav2_accuracy']*100:6.2f}%   ({(v3e['vqav2_accuracy']-v3b['vqav2_accuracy'])*100:+.2f}%p)")
    print(f"  POPE acc: v2 {v2['pope_accuracy']*100:6.2f}%  →  v3-base {v3b['pope_accuracy']*100:6.2f}%  →  v3-ENH {v3e['pope_accuracy']*100:6.2f}%   ({(v3e['pope_accuracy']-v3b['pope_accuracy'])*100:+.2f}%p)")
    print(f"  POPE F1:  v2 {v2['pope_yes_f1']:6.3f}   →  v3-base {v3b['pope_yes_f1']:6.3f}   →  v3-ENH {v3e['pope_yes_f1']:6.3f}    ({v3e['pope_yes_f1']-v3b['pope_yes_f1']:+.3f})")
    print(f"\n  paths: {paths}")


if __name__ == "__main__":
    main()
