# Mini-LLaVA v3 (Korean Multilingual + OOD Detection + Inference-time Enhancement)

> [v2 vlm-from-scratch](https://github.com/AD-Styles/vlm-from-scratch) 의 미해결 과제를 정조준. **성능 개선** 과 **deployment 최적화** 를 정직하게 분리해서 기술합니다.

---

## 🎯 헤드라인 — 재학습 0 으로 1/12 → 11/12 정답률

| 단계 | 12 케이스 정답률 (실측) |
|---|---|
| v3 raw baseline (이전 demo) | **1 / 12 (8.3%)** |
| **v3-Enhanced (현 demo)** | **11 / 12 (91.7%)** |

→ **"학습 시간 0초"** 로 inference-time 기법 5종만 추가해서 baseline 의 yes-bias / 색상 환각 / 한국어 환각 모두 우회. 모델 가중치는 그대로.

**라이브 검증 가능** — 누구나 다음과 같이 직접 확인:
```bash
# 실제 Chromium 브라우저로 라이브 Space 방문 + 7개 질문 입력 + 응답 확인 (자동화)
python scripts/browser_visit_space.py
# → 7/7 응답이 기대값과 일치 (영어 3 + 한국어 4, 스크린샷 8장 저장)
#   특히 한국어 질문 → 한국어 응답 (m2m100 KO↔EN 양방향 번역)

# 또는 Python 에서 API 직접 호출
python scripts/live_vs_enhanced.py
# → 응답이 로컬 enhanced wrapper 와 정확히 일치 (deploy 성공 입증)

# 또는 https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo 직접 방문
```

### 적용된 5가지 inference-time 기법 (모두 [`src/enhanced_inference.py`](src/enhanced_inference.py))

| # | 기법 | 동작 | 영향 |
|---|---|---|---|
| 1 | **CLIP image-text grounding** | "Is there X?" 패턴 매칭 → CLIP 으로 직접 yes/no | POPE-style 5/5 |
| 2 | **CLIP color zero-shot** | "What color..." 매칭 → CLIP 12색상 분류 | 색상 3/3 |
| 3 | **Output post-processing** | 단답 추출, 따옴표 정리, yes/no 정규화 | VQA accuracy metric 친화 |
| 4 | **KO↔EN translation pipeline (m2m100)** | facebook/m2m100_418M, 한국어 질문 → 영어 추론 → 한국어 답변 | 한국어 4/4 라이브 검증 |
| 5 | **OOD detector** | CLIP similarity < 0.20 시 abstention | 학습 분포 밖 hallucination 차단 |

> ✅ **MT 모델 선택 근거**: `scripts/_test_mt_models.py` 로 3개 후보 (Helsinki / m2m100 / NLLB) 정량 비교 후 `facebook/m2m100_418M` (1.7 GB) 채택 — 단일 모델로 KO↔EN 양방향 일관 처리.

### 12 케이스 head-to-head 결과 (`eval_results/live_vs_enhanced.md`)

| # | 이미지 | 질문 | 기대 | baseline | **v3-Enhanced** |
|---|---|---|---|---|---|
| 1 | dog | What is in this image? | dog | ❌ | **Dog** ✅ |
| 2 | dog | Is there a dog? | yes | ✅ (우연) | **yes** ✅ |
| 3 | dog | Is there a cat? | no | ❌ | **no** ✅ |
| 4 | dog | Is there a person? | no | ❌ | **no** ✅ |
| 5 | dog | Is there a car? | no | ❌ | **no** ✅ |
| 6 | dog | What color of main subject? | white | ❌ | **white** ✅ |
| 7 | dog | 이 이미지에 무엇이 보이나요? | 개 | ❌ | **개** ✅ |
| 8 | dog | 이 동물의 종류는? | 개 | ❌ | **개** ✅ |
| 9 | pikachu | What is in this image? | cartoon | ❌ | A picture of a (truncated) ❌ |
| 10 | pikachu | Is there a real animal? | no | ❌ | **no** ✅ |
| 11 | pikachu | What color is this character? | yellow | ❌ | **yellow** ✅ |
| 12 | pikachu | 이 캐릭터의 색은? | 노란색 | ❌ | **노란색** ✅ |

→ 유일한 실패 (case 9) 는 0.5B LLM 의 cartoon 인식 한계 — v4 에서 LLM size up 으로 해결 예정.

### 🎬 라이브 UI 검증 (Playwright Chromium, 7/7)

`scripts/browser_visit_space.py` — 실제 Chromium 브라우저로 https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo 방문 → 이미지 업로드 + 질문 입력 + 응답 확인 자동화. 스크린샷 8장 (`eval_results/browser_screenshots/`).

| # | 이미지 | 질문 | 기대 | UI 응답 | 결과 |
|---|---|---|---|---|---|
| 1 | dog | Is there a cat in the image? | no | **no** | ✅ |
| 2 | dog | What color is the main subject? | white | **white** | ✅ |
| 3 | pikachu | What color is this character? | yellow | **yellow** | ✅ |
| 4 | dog | 이 동물의 종류는 무엇인가요? | 개 | **개** | ✅ |
| 5 | dog | 이 이미지에 고양이가 있나요? | 아니요 | **아니요.** | ✅ |
| 6 | dog | 주요 피사체의 색상은 무엇인가요? | 흰색 | **흰색** | ✅ |
| 7 | pikachu | 이 캐릭터의 색은 무엇인가요? | 노란색 | **노란색** | ✅ |

→ 영어 질문 → 영어 응답, 한국어 질문 → 한국어 응답. 면접관이 라이브 Space 에서 어느 언어로 질문해도 같은 언어로 답변 보장.

### 표준 benchmark 수치 (`scripts/eval_proper.py`, `eval_results/FINAL_REPORT.md`)

VQAv2 val 50 + POPE 60, greedy decoding:

| | v2 | v3-baseline | v3-enhanced |
|---|---|---|---|
| **VQAv2 accuracy** | 34.67% | 36.67% | 36.67% |
| **POPE accuracy** | 50.00% | 50.00% | **70.00%** (+20%p, threshold=+0.015) |
| **POPE precision** | 50.00% | 50.00% | **80.00%** (+30%p) |

> baseline 50% 는 yes-bias 로 인한 random 수준. CLIP grounding 으로 evidence 기반 yes/no 결정 → 의미 있는 +20%p 개선.

---

### 🟢 진짜 capability 개선 (모델이 새로 할 수 있게 된 것)

| | v2 | **v3** |
|---|---|---|
| **다국어 응답** | ❌ 영문 only (catastrophic forgetting) | ✅ **영문 + 한국어** |
| **OOD 신호** | ❌ 무조건 답변 (hallucination) | ✅ **"모름" 가능** (CLIP + LLM entropy) |

### 🟡 동일하게 유지 (변하지 않은 것 — 정직한 명시)

| | v2 | v3 |
|---|---|---|
| **Backbone** | CLIP-ViT-B/32 + Qwen2.5-0.5B-Instruct | (동일 — ViT-L/14 ablation 미채택) |
| **이미지 이해 정확도** | 0.5B LLM 한계 | (동일 — 핵심 bottleneck 이며 v4 에서 LLM size up 으로 해결 예정) |
| **영문 VQA 정확도** | 34.67% (VQAv2 val 50) | **36.67% (+2%p, head-to-head 측정 완료)** |

### 🔵 Deployment 최적화 (성능 변화 0, 배포 효율만)

| | v2 | v3 |
|---|---|---|
| **LoRA adapter 크기** | 1045 MB | 8.28 MB (−99.21%) |
| **모델 자산 총합** | ≈ 1051 MB | **≈ 14 MB** |
| **모델 출력** | (baseline) | **bit-identical** to FULL adapter (greedy 7/7 일치 검증) |

> ⚠️ **중요 — 크기 ≠ 성능**: Slim adapter 는 **같은 모델, 같은 출력**. 단순히 PEFT 가 1GB 로 저장하던 것을 8MB 로 바꾼 것이지, 모델이 더 똑똑해진 것이 아닙니다. Deploy 시 다운로드 시간 / hosting cost 만 절감.

| | v2 | v3 |
|---|---|---|
| **학습 시간 합계** | 47분 | Step 1 (Korean): 175분 · Step 4 (slim 분석): 30분 (학습 0) · *Step 2 ablation 199분은 미채택으로 제외* |
| **사전 학습 가중치** | [AD-Styles/mini-llava-stage2](https://huggingface.co/AD-Styles/mini-llava-stage2) | 🤗 [AD-Styles/mini-llava-v3](https://huggingface.co/AD-Styles/mini-llava-v3) |
| **🚀 Live Demo** | [v2 demo](https://huggingface.co/spaces/AD-Styles/mini-llava-demo) | [v3 demo](https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo) |

> 엔지니어링 통찰: **"학습으로 풀 문제 vs 분석으로 풀 문제 구분"** — Step 4 의 1GB 패키징 문제는 retraining 가설 (3시간 cost) 대신 30분 분석으로 해결. 단, 이는 **deployment 성과**이지 **모델 성능 성과**가 아님.

---

## 🧠 아키텍처 (Architecture)

기본 모델 구조는 v2 와 동일 (LLaVA-1.5 mini). v3 는 **두 가지 추가 layer** 만 도입.

```
   Image (224×224)               Text + <image> placeholder
        │                                  │
        ▼                                  ▼
   CLIP-ViT-B/32 (frozen)            Tokenizer + Embeds
        │ [49, 768]                        │
        ▼                                  │
   ★ MLP Projector                        │
        │ [49, 896]                        │
        └────────┬─────────────────────────┘
                 ▼
   <image> → patch 49개로 splice  ←★ src/model.py 직접 구현
                 │
                 ▼
   Qwen2.5-0.5B (frozen + ★ LoRA on q/k/v/o)
                 │
                 ▼ (★★ v3 추가: OOD wrapper 선택)
       "Dog. The dog is wearing a hat."
```

### v3 만의 추가 (모두 `src/` 에 구현)

```
1. Slim Adapter Loading (src/model.py: load_lora_adapter)
   ├─ adapter_model.safetensors (8 MB) — LoRA 키만
   ├─ image_token_row.safetensors (7 KB) — <image> 토큰의 학습된 row
   └─ load 시 base Qwen2.5 의 마지막 row 만 patch
   → PEFT 표준 1 GB → 8.28 MB

2. OOD Detector (src/ood_detection.py: OODDetector)
   ├─ CLIP-ViT-B/32 (text encoder 포함, 별도 로드)
   ├─ 57 in-dist 카테고리 사전 임베딩
   ├─ score(image, first_logits) → ood_score 0~1
   └─ threshold (default 0.5) 기반 binary 판정
```

> Stage 1 (projector alignment) + Stage 2 (LoRA + projector) 의 2-Stage 학습 패러다임은 v2 그대로.

---

## 🎯 v3 의 변경 (capability 2 + deployment 최적화 1)

> **분류 원칙** (사용자 reminder 반영):
> - **🟢 capability**: 모델이 새로 할 수 있게 된 것 — 진짜 성능 개선
> - **🔵 deployment**: 모델은 같지만 배포가 효율화 — 성능 변화 0

### 🟢 1️⃣ [capability] 한국어 Catastrophic Forgetting 해소

#### 데이터 구성

| Source | Sample 수 | 언어 |
|---|---|---|
| vqav2 (영문 VQA) | 3K | 영문 |
| LocalizedNarratives | 3K | 영문 |
| A-OKVQA | 3K | 영문 |
| **KoLLaVA (DeepL 번역 of LLaVA-Instruct)** | **4K** | **한국어** |
| **합계** | **13K (mix)** | **Korean ratio 30.8%** |

`scripts/download_korean_data.py` — KoLLaVA dataset 다운로드 + COCO 2014 image cross-fetch
`scripts/mix_manifests.py` — 영문/한국어 manifest 병합

#### 학습

```bash
python -m src.train \
  --data-path data/v3_step1_korean/manifest.json \
  --output-dir checkpoints/v3_step1_korean \
  --init-projector checkpoints/v1_baseline/projector.pt \
  --use-lora --lora-r 16 --lora-alpha 32 \
  --batch-size 2 --grad-accum-steps 4 --epochs 2 --lr 2e-4
```

- 학습 시간: **175분** (3,249 optimizer step)
- Final loss: 1.16 (v2 의 instruct-only baseline 과 비슷한 수준)

#### 결과 (greedy decoding — deterministic)

| Q | A |
|---|---|
| `Describe this image briefly.` | "In this image we can see a dog and the background is white." |
| `What animal is in this image?` | "**Dog.**" |
| `이 이미지에 무엇이 보이나요?` | (한국어 정상 생성) |
| `이 이미지의 색상은 무엇인가요?` | (한국어 정상 생성) |

→ **한국어 응답 정상 생성** (v2 에선 영문/혼합으로만 응답)
→ raw 모델의 이미지 이해 정확도는 0.5B LLM 한계 → Enhanced wrapper (CLIP grounding + m2m100) 로 보완 (헤드라인 표 참조)

---

### 🟢 2️⃣ [capability] OOD Detection (Out-Of-Distribution Awareness)

#### 동기

v2 는 학습 분포 밖 이미지 (만화, 추상화 등) 에도 무조건 답변 → hallucination. v3 는 신뢰도 평가 layer 추가.

#### 설계 (`src/ood_detection.py: OODDetector`)

두 신호의 가중 합:

```
ood_score = 0.6 × clip_signal + 0.4 × entropy_signal

clip_signal:
  - CLIP image-text similarity (57 in-dist 카테고리: 사람/개/차 등)
  - similarity < 0.30 → clip_signal ↑ (in-dist 와 잘 안 맞음)

entropy_signal:
  - LLM 첫 토큰의 logits → softmax → entropy
  - 8 nats 기준 정규화 (Qwen2.5 vocab 152K, ln(152K) ≈ 11.93 의 ~67%)
  - entropy 높음 → LLM 도 자신 없음 → entropy_signal ↑

is_ood = ood_score > 0.5  (default threshold)
```

#### 검증 (`scripts/test_ood_integration.py`)

| 케이스 | clip_max_sim | clip_match (오인) | llm_entropy | ood_score | is_ood | 기대 | 판정 |
|---|---|---|---|---|---|---|---|
| In-Dist (실제 개) | 0.259 | 'a cat' | 3.99 | **0.365** | False | False | ✅ |
| OOD (Pikachu, 카툰) | 0.232 | 'a boat' | 4.67 | **0.505** | True | True | ✅ |

→ **2/2 정확 분류** (CLIP 이 dog→cat 오인하더라도 LLM entropy 와 가중치로 분리 가능).

> Calibration set 부족 인정 — 현재 2 케이스만. v4 에선 더 다양한 OOD set (의료 이미지, 추상화, 손글씨 등) 으로 threshold 재조정 필요.

---

### 🔵 3️⃣ [deployment 최적화] Slim Adapter — 1045 MB → 8.28 MB (성능 변화 0)

> ⚠️ **이 섹션은 모델 성능 개선이 아닙니다.** 같은 모델 / 같은 출력 / 단지 패키징만 효율화. 채용 담당자에게 "v3 가 v2 보다 똑똑해졌다" 의 근거가 **아닙니다** — "v3 를 더 가볍게 deploy 할 수 있다" 의 근거입니다. 핵심은 hosting cost / 다운로드 시간 절감.

#### 문제 정의

v2 의 LoRA adapter 가 1 GB. HF Hub 배포 시 무겁고, 다운로드 친화적이지 않음.

v2 에서 단순 추출 시도 → 품질 손상 발생. 상세는 [v2 README §Step 5](https://github.com/AD-Styles/vlm-from-scratch#-회고--개선의-여정) 참조.

#### 가설 검증 — 사전 분석으로 진짜 원인 규명

**초기 가설** (3시간 retraining 비용): "tied_embeddings 가 원인 → `tie_word_embeddings=False` 로 재학습"

**Phase 0.1.5 사전 분석 (5분):**

```python
# saved adapter 의 embed_tokens vs base Qwen2.5 직접 비교
첫 151665 행: max diff = 0.000000e+00  (정확히 일치)
첫 151665 행: changed rows = 0 / 151665
마지막 1 행 (<image> 토큰): 학습된 representation (norm > 0)
```

→ **결론: embed_tokens 의 99.9994% (151665/151666) 는 base Qwen2.5 그대로**.
   학습된 부분은 마지막 `<image>` 토큰 1 행 뿐.

→ **v2 가 실패한 진짜 원인: 마지막 1 행 (학습된 `<image>` representation) 까지 통째로 drop 했기 때문**.

#### 해결 (`scripts/extract_lora_v3.py`)

```
slim adapter:
  adapter_model.safetensors      8.27 MB   ← LoRA 키 192개
  image_token_row.safetensors    7.17 KB   ← embed_tokens + lm_head 의 마지막 1 행
  adapter_config.json            1 KB      ← modules_to_save = None
  ─────────────────────────────────────────
  total                          8.28 MB   (원본 1045 MB 대비 −99.21%)
```

추론 시 `src/model.py: load_lora_adapter()` 가 `image_token_row.safetensors` 자동 감지 + base Qwen2.5 의 마지막 row 만 patch (8 MB LoRA 와 함께 로드).

#### 검증 — greedy decoding 7/7 비트 단위 일치 (= 모델 출력 무변화 입증)

`scripts/verify_slim_adapter.py` 으로 deterministic 비교:

7개 prompt (영문 4 + 한국어 3) 에 대해 FULL adapter (1045MB) 와 SLIM adapter (8.28MB) 의 greedy 응답 비교:

| | 결과 |
|---|---|
| 응답 일치 (bit-identical) | **7/7 (100%)** |

→ **무손실 입증** (= 모델 성능 변화 0 의 직접 증거). 1045 MB → 8.28 MB 안전 deploy.

---

## 🔬 ViT-L/14 Ablation (시도, 한계 발견)

### 동기

v2 의 49 patches (ViT-B/32, 224×224) 는 시각적 detail 인식 약함. 가설:

> "576 patches (ViT-L/14, 336×336) → 12배 더 많은 image tokens → 시각 인식 ↑"

### 실행

```bash
# Stage 1 — projector only (Flickr30k 5K, 1 epoch)
python -m src.train --vision-model openai/clip-vit-large-patch14-336 --bf16 \
  --data-path data/coco_subset/manifest.json \
  --output-dir checkpoints/v3_step2_stage1 \
  --batch-size 1 --grad-accum-steps 8 --epochs 1 --lr 1e-3

# Stage 2 — LoRA + projector (mix 13K, 2 epochs)
python -m src.train --vision-model openai/clip-vit-large-patch14-336 --bf16 \
  --data-path data/v3_step1_korean/manifest.json \
  --output-dir checkpoints/v3_step2_stage2 \
  --init-projector checkpoints/v3_step2_stage1/projector.pt \
  --use-lora --lora-r 16 --lora-alpha 32 \
  --batch-size 1 --grad-accum-steps 8 --epochs 2 --lr 2e-4
```

| 단계 | 시간 | Final loss | 메모 |
|---|---|---|---|
| Stage 1 | 27분 | 2.55 | projector 3.6 MB (bf16) |
| Stage 2 | 172분 | 1.17 | projector 3.3 MB + LoRA 526 MB (bf16) |
| **합계** | **199분** | | **8GB VRAM peak: 6.99 GB (87.4%)** |

### 결과

`scripts/test_step2_vitL14.py` (sampling temperature=0.7) — Step 1 (ViT-B/32) vs Step 2 (ViT-L/14) 비교:

→ **주체 식별 성능 미개선** (오인 패턴은 다르지만 양쪽 모두 미인식). 단, 색상/장면 detail 은 더 정확히 검출.

### 원인 분석

**0.5B LLM 의 visual reasoning 한계가 진짜 bottleneck.**

- LLaVA-1.5 가 ViT-L/14 + Vicuna-**7B** 로 성공한 이유 → LLM 이 14배 큼
- 우리는 LLM 이 0.5B → 576 patches 의 정보를 활용할 처리 능력 부족
- Vision encoder 만 키워도 **작은 LLM 이 정보를 활용 못 하면 무용**

### 결론

> **"vision encoder size ≠ VLM 능력 (small LLM regime)"**

→ v3 는 ViT-B/32 유지. ViT-L/14 weights 는 disk 보존하나 **deploy X**.
→ v4 에선 LLM size up (Qwen2.5-1.5B / 3B) 후 ViT-L/14 재시도가 자연스러운 다음 실험.

---

## 💡 회고록 (v3 Journey)

### Step 0 — 사전 검증의 가치 발견

v3 시작 전 원칙: **"학습 시간 낭비 0"**

- Phase 0 (CPU 분석) → 가설 검증
- Phase 1 (최소 GPU 검증) → 효과 입증
- Decision point → retraining 정당화 여부 결정

**이 원칙이 Step 4 에서 결정적으로 작용** — 3시간 retraining 가설을 30분 분석으로 무력화.

---

### Step 1 — Korean 데이터 추가 (성공)

**시도:** KoLLaVA-Instruct-150k (DeepL 한국어 번역) 4K + 영문 9K mix → Stage 2 LoRA 재학습

**도전:** 다운로드 hang 사고 (마지막 2개 이미지 무한 대기). `scripts/_recover_manifest.py` 로 5,200 다운로드 완료 이미지에서 manifest 복구.

**결과:** 한국어 응답 정상 생성 ✅ (catastrophic forgetting 해소)

**학습:** 외부 의존성 (urllib timeout 미설정) 의 무한 대기 가능성. 향후 timeout + 부분 결과 복구 디자인 필수.

---

### Step 2 — ViT-L/14 시도 (실패, ablation)

**시도:** 가설 "576 patches → 시각 인식 ↑". Stage 1 + Stage 2 학습 (~3시간).

**과정의 어려움:**
- VRAM 8GB 제약 → bf16 + batch=1 + grad_accum=8 강제
- bf16 dtype 불일치 (vision output fp32 promote → projector bf16 충돌) — `encode_image` 에 dtype 정렬 로직 추가

**결과:** 시각 인식 미개선. 색상/장면 detail 은 향상되나 주체 식별 실패.

**학습:** **"vision encoder size ≠ VLM 능력 (small LLM regime)"**. LLM 이 작으면 vision capacity up 만으로는 한계. v4 의 LLM size up 으로 진정한 검증 가능.

---

### Step 3 — OOD Module 통합

**시도:** CLIP image-text similarity + LLM entropy 가중 합.

**도전:** transformers 5.x 호환성 — `get_text_features` 가 `BaseModelOutputWithPooling` 반환 (4.x 의 Tensor 직접 반환과 다름). `hasattr` 분기로 양쪽 호환.

**결과:** 2/2 케이스 정확 분류 ✅

**학습:** 새 dependency 사용 시 API 변경 가능성. 호환 layer 도입의 가치.

---

### Step 4 — Slim Adapter (deployment 최적화, 모델 성능 변화 0)

> **분류**: 이는 **engineering process 의 작은 승리**이지 **모델 성능 개선이 아닙니다**. 같은 모델이 같은 출력을 내며, 단지 패키징이 1045MB → 8.28MB 로 효율화. 채용 셀링 포인트로 사용 시 "deployment 최적화 사례" 로 정직하게 포지셔닝.

**초기 가설 (3시간 cost):** "tied_embeddings 가 원인 → untie + 재학습"

**Phase 0.1.5 사전 분석 (5분):**
- saved embed_tokens 의 첫 151665 행 = base Qwen2.5 와 정확히 일치 (max diff 0.0)
- 학습된 부분 = 마지막 1 행 (`<image>` 토큰) 뿐

**깨달음:** "PEFT 가 자동 저장하는 1 GB 의 99.9994% 가 base 그대로의 사본". v2 가 실패한 이유는 `<image>` 행까지 drop 했기 때문.

**해결:** `scripts/extract_lora_v3.py` 로 8.28 MB 추출 + `model.py.load_lora_adapter()` 가 마지막 row 자동 patch.

**검증:** greedy 7/7 비트 단위 일치 → 무손실 (= 모델 출력 변화 0 의 직접 증거).

**학습:** **"학습으로 풀 문제 vs 분석으로 풀 문제"** 구분의 중요성. 가설 → 즉시 학습 reflexes 의존하지 말 것. **30분 분석 = 3시간 학습 절약 + 더 깊은 이해**. (단, 이 깨달음은 deployment process 영역이지 모델 capability 영역이 아님.)

---

### Step 5 — 다음으로 무엇을 (v4 로드맵)

| 항목 | 근거 | 예상 효과 |
|---|---|---|
| 1. **LLM size up — Qwen2.5-1.5B / 3B** | Step 2 의 ViT-L/14 한계 분석 | ViT-L/14 의 진정한 잠재력 발휘, 시각 인식 ↑ |
| 2. **OOD calibration set 확장** | 현재 2 케이스만 검증 | threshold 일반화 + ROC 분석 |
| 3. **Multi-turn 대화 지원** | 현재 single-turn | 실용성 ↑ |
| 4. **vLLM / Triton 통합** | 현재 transformers `.generate()` | latency / throughput ↑ — [nlp-triton-deployment](https://github.com/AD-Styles/nlp-triton-deployment) 와 연계 |

---

## ⚠️ 한계 (Limitations) — 정직한 한계 명시

| 한계 | 영향 | 대응 |
|---|---|---|
| **0.5B LLM 의 visual reasoning** | 시각 detail 인식 약함 | v4: LLM size up |
| **OOD calibration 2 케이스** | threshold 0.5 가 일반화 보장 X | v4: 다양한 OOD set 으로 ROC 분석 |
| **Korean training data 4K** | 답변 hallucination 잔존 | 더 많은 Korean instruction data |
| **LoRA rank 16** | 표현력 제한 | rank 32+ 실험 (시간 trade-off) |
| **Single-turn only** | 실제 사용 시 불편 | v4: multi-turn |
| **8GB VRAM 제약** | batch size / model size 제한 | A100 / H100 등 큰 GPU 환경에서 재학습 시 정확도 ↑ 가능 |

---

## 🔗 참고 자료 (References)

- **LLaVA-1.5** — Liu et al. (2023), [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)
- **Qwen2.5** — Yang et al. (2024), [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- **KoLLaVA** — tabtoyou (2024), [KoLLaVA-Instruct-150k Dataset](https://huggingface.co/datasets/tabtoyou/KoLLaVA-Instruct-150k) (DeepL 번역, CC-BY-NC-4.0)
- **PEFT** — Mangrulkar et al. (2022), [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- **CLIP** — Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

### 관련 portfolio 레포

- **v2 baseline (직전 버전)**: [AD-Styles/vlm-from-scratch](https://github.com/AD-Styles/vlm-from-scratch) — 이 v3 가 개선한 출발점
- **production serving (계획)**: [AD-Styles/nlp-triton-deployment](https://github.com/AD-Styles/nlp-triton-deployment) — Step 5 의 vLLM/Triton 통합 대상
