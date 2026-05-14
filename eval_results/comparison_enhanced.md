# 최종 성능 비교 — v2 vs v3-baseline vs v3-enhanced

평가: VQAv2 val 50 + POPE test 60, greedy decoding.

> ℹ️ **POPE threshold 주석**: 이 표는 demo deploy 와 동일한 `pope_threshold=0.0` (관대) 기준 — POPE 53.33%.
> POPE benchmark 최적값 `+0.015` 으로 재계산 시 70% (+20%p), precision 80% (+30%p) — `eval_results/FINAL_REPORT.md` 참조.

## VQAv2 (공식 VQA accuracy)

| 항목 | v2 | v3-baseline | v3-enhanced | enhanced - baseline |
|---|---|---|---|---|
| 전체 | 34.67% | 36.67% | **36.67%** | +0.00%p |
| answer_type='number' | 0.00% | 11.11% | **11.11%** | +0.00%p |
| answer_type='other' | 28.07% | 28.07% | **28.07%** | +0.00%p |
| answer_type='yes/no' | 54.55% | 54.55% | **54.55%** | +0.00%p |

## POPE (hallucination 평가)

| Metric | v2 | v3-baseline | v3-enhanced | enhanced - baseline |
|---|---|---|---|---|
| 전체 정확도 | 50.00% | 50.00% | **53.33%** | +3.33%p |
| yes-Recall | 100.00% | 100.00% | **80.00%** | -20.00%p |
| yes-Precision | 50.00% | 50.00% | **52.17%** | +2.17%p |
| yes-F1 | 0.667 | 0.667 | **0.632** | -0.035 |

## V3-Enhanced 라우팅 통계 (VQAv2 50 샘플)

| Path | 사용 횟수 |
|---|---|
| vlm_raw | 46 |
| clip_color | 3 |
| clip_grounding_yesno | 1 |