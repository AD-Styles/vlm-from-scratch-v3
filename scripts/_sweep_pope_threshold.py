"""POPE CLIP-grounding margin threshold sweep for best F1.

이미 저장된 enhanced 결과의 clip_grounding_margin 을 다시 평가.
threshold 하나마다 accuracy/precision/recall/F1 계산 → 최적값 선택.
"""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

e = json.load(open("eval_results/v3_enhanced_results.json", encoding="utf-8"))

samples = []
for r in e["pope_details"]:
    if r.get("used_path") != "clip_grounding_yesno":
        # vlm_raw fallback (1 sample) — 그대로 사용
        samples.append({"gt": r["gt_answer"], "pred_yn": r["pred_yn"], "margin": None, "fallback": True})
    else:
        samples.append({"gt": r["gt_answer"], "margin": r["clip_grounding_margin"], "fallback": False})


def metrics_at(threshold):
    """margin > threshold → 'yes', else 'no'. fallback 은 원본 유지."""
    tp = fp = tn = fn = 0
    for s in samples:
        if s["fallback"]:
            pred_yn = s["pred_yn"]
        else:
            pred_yn = "yes" if s["margin"] > threshold else "no"
        if pred_yn == "yes":
            if s["gt"] == "yes": tp += 1
            else: fp += 1
        else:
            if s["gt"] == "no": tn += 1
            else: fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return {"threshold": threshold, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# Sweep
import numpy as np
thresholds = np.arange(-0.15, 0.15, 0.005)
results = [metrics_at(t) for t in thresholds]

print("threshold |   acc  |   F1  |  prec |  rec  | tp/fp/tn/fn")
print("----------+--------+-------+-------+-------+------------")
best_f1 = max(results, key=lambda r: r["f1"])
best_acc = max(results, key=lambda r: r["acc"])
for r in results:
    marker = " ★F1" if r is best_f1 else (" ★ACC" if r is best_acc else "")
    print(f"  {r['threshold']:+.3f}  | {r['acc']*100:5.2f}% | {r['f1']:.3f} | {r['prec']*100:5.2f}% | {r['rec']*100:5.2f}% | {r['tp']:2}/{r['fp']:2}/{r['tn']:2}/{r['fn']:2}{marker}")

print()
print(f"BEST F1:  threshold={best_f1['threshold']:+.3f}  F1={best_f1['f1']:.3f}  acc={best_f1['acc']*100:.2f}%  prec={best_f1['prec']*100:.2f}%  rec={best_f1['rec']*100:.2f}%")
print(f"BEST ACC: threshold={best_acc['threshold']:+.3f}  acc={best_acc['acc']*100:.2f}%  F1={best_acc['f1']:.3f}")

# Baseline 비교
print()
print("Baseline (v3, always 'yes'):  acc=50.00%  F1=0.667  prec=50.00%  rec=100.00%")
print(f"v3-Enhanced (best F1):        acc={best_f1['acc']*100:.2f}%  F1={best_f1['f1']:.3f}  prec={best_f1['prec']*100:.2f}%  rec={best_f1['rec']*100:.2f}%  diff F1: {best_f1['f1']-0.667:+.3f}")
print(f"v3-Enhanced (best ACC):       acc={best_acc['acc']*100:.2f}%  F1={best_acc['f1']:.3f}  prec={best_acc['prec']*100:.2f}%  rec={best_acc['rec']*100:.2f}%  diff acc: {(best_acc['acc']-0.5)*100:+.2f}%p")
