# v2 vs v3 정직한 성능 비교

## 평가 설정

- VQAv2 val: 50 samples (lmms-lab/VQAv2, streaming)
- POPE test: 60 samples (lmms-lab/POPE, streaming, COCO val2014 기반)
- decoding: greedy (do_sample=False) — deterministic
- v2: AD-Styles/mini-llava-stage2 (1045MB adapter)
- v3: AD-Styles/mini-llava-v3 (8.28MB slim adapter, Korean training)

## VQAv2 — 공식 VQA accuracy

| 항목 | v2 | v3 | diff |
|---|---|---|---|
| 전체 | 34.67% | 36.67% | +2.00%p |
| answer_type='number' | 0.00% | 11.11% | +11.11%p |
| answer_type='other' | 28.07% | 28.07% | +0.00%p |
| answer_type='yes/no' | 54.55% | 54.55% | +0.00%p |

## POPE — hallucination 평가

POPE = Polling-based Object Probing Evaluation. yes/no 단답 평가 데이터셋.

| Metric | v2 | v3 | diff |
|---|---|---|---|
| 전체 정확도 | 50.00% | 50.00% | +0.00%p |
| yes-Recall | 100.00% | 100.00% | +0.00%p |
| yes-Precision | 50.00% | 50.00% | +0.00%p |
| yes-F1 | 0.667 | 0.667 | +0.000 |
| Refusal rate (?) | 0.00% | 0.00% | +0.00%p |

### POPE — category 별 정확도

| category | v2 | v3 | diff |
|---|---|---|---|
| adversarial | 50.00% | 50.00% | +0.00%p |