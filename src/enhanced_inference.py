"""V3-Enhanced inference wrapper — 재학습 없이 추론 시점 성능 개선.

통합 기법:
  1. Output extraction        — VQA 단답 추출, yes/no 정규화, 따옴표/구두점 정리
  2. Constrained generation   — 질문 type 별 max_new 차등 (yesno=3, vqa=10, describe=64)
                                + repetition_penalty (한국어 rambling 차단)
  4. Ko→En→Ko 번역 파이프라인 — Helsinki-NLP MT 로 한국어를 영문 추론 라인 위에 매핑
  6. OOD-gated abstention      — OODDetector 점수 > threshold 시 "모름" 응답 (hallucinate 대신)
  7. CLIP zero-shot subject    — what/which 질문일 때 CLIP 이 직접 분류한 결과를 v3 응답과 cross-check

설계 결정:
  - 모델 가중치는 절대 수정 X. 학습 0.
  - OODDetector 의 CLIP 모델을 재사용 (모델 1개로 OOD + 보조 분류 둘 다)
  - 번역 모델 (KO→EN, EN→KO) 은 첫 사용 시 lazy load (~600 MB)
"""
from __future__ import annotations

import re
from typing import Optional

import torch
from PIL import Image

from .dataset import encode_for_inference
from .model import MiniLLaVA
from .ood_detection import OODDetector


# 질문 type 별 generation 파라미터
QTYPE_CONFIG = {
    "yesno":     {"max_new": 5,   "rep_penalty": 1.0},
    "vqa":       {"max_new": 32,  "rep_penalty": 1.0},   # 16→32 (case 9 truncation fix)
    "color":     {"max_new": 8,   "rep_penalty": 1.0},
    "count":     {"max_new": 5,   "rep_penalty": 1.0},
    "describe":  {"max_new": 64,  "rep_penalty": 1.3},
    "default":   {"max_new": 64,  "rep_penalty": 1.1},
    "korean":    {"max_new": 128, "rep_penalty": 1.2},
}

# CLIP 보조 분류용 카테고리 (COCO 80 + 자주 등장 객체)
CLIP_CLASSES_PROMPTS = [
    "a photo of a person", "a photo of a man", "a photo of a woman", "a photo of a child",
    "a photo of a dog", "a photo of a cat", "a photo of a horse", "a photo of a cow",
    "a photo of a sheep", "a photo of a bird", "a photo of an elephant", "a photo of a bear",
    "a photo of a zebra", "a photo of a giraffe",
    "a photo of a car", "a photo of a truck", "a photo of a bus", "a photo of a train",
    "a photo of a motorcycle", "a photo of a bicycle", "a photo of an airplane", "a photo of a boat",
    "a photo of a chair", "a photo of a couch", "a photo of a bed", "a photo of a tv",
    "a photo of a laptop", "a photo of a book", "a photo of a cup", "a photo of a bottle",
    "a photo of a pizza", "a photo of a sandwich", "a photo of a cake",
    "a photo of a banana", "a photo of an apple", "a photo of an orange",
    "a photo of a building", "a photo of a tree", "a photo of a mountain",
    "a photo of a beach", "a photo of a road",
    "a photo of a kitchen", "a photo of a living room", "a photo of a bathroom",
]
# CLIP 분류 → 한 단어 라벨 매핑
CLIP_LABEL = [p.replace("a photo of a ", "").replace("a photo of an ", "") for p in CLIP_CLASSES_PROMPTS]


# ──────────────────────────────────────────────────────────────────
# 기법 1: Output extraction
# ──────────────────────────────────────────────────────────────────
def extract_yesno(text: str) -> str:
    """첫 yes/no 토큰 추출. 못 찾으면 원본의 첫 5단어 반환."""
    t = text.lower().strip()
    head = t[:30]
    # yes 가 더 앞에 있으면 yes, no 가 더 앞에 있으면 no
    yes_pos = head.find("yes")
    no_pos = head.find("no")
    if yes_pos == -1 and no_pos == -1:
        return text.strip().split()[0] if text.strip() else "?"
    if yes_pos == -1:
        return "no"
    if no_pos == -1:
        return "yes"
    return "yes" if yes_pos < no_pos else "no"


def extract_short_answer(text: str) -> str:
    """VQA 단답 추출: 첫 문장 → 첫 명사 phrase 까지."""
    t = text.strip()
    # 첫 문장
    for sep in [".", "?", "!", "\n"]:
        if sep in t:
            t = t.split(sep)[0]
    # 따옴표 제거
    t = t.strip().strip("'\"").strip()
    # 너무 길면 첫 4 단어로 자름 (VQA accuracy metric 은 단답 위주)
    words = t.split()
    if len(words) > 4:
        t = " ".join(words[:4])
    return t


# ──────────────────────────────────────────────────────────────────
# 질문 type 분류
# ──────────────────────────────────────────────────────────────────
YESNO_PATTERNS = [
    r"\bis\b", r"\bare\b", r"\bdoes\b", r"\bdo\b", r"\bcan\b",
    r"\bhas\b", r"\bhave\b", r"\bwill\b", r"\bwas\b", r"\bwere\b",
]
COLOR_KEYS = ["color", "colour", "what color", "which color"]
COUNT_KEYS = ["how many", "count", "number of"]
DESCRIBE_KEYS = ["describe", "explain", "tell me about", "what do you see", "설명", "묘사"]


def detect_question_type(question: str) -> str:
    q = question.lower().strip()
    if any(k in q for k in DESCRIBE_KEYS):
        return "describe"
    if any(k in q for k in COUNT_KEYS):
        return "count"
    if any(k in q for k in COLOR_KEYS):
        return "color"
    # yes/no 패턴
    first_word = q.split()[0] if q.split() else ""
    if first_word in {"is", "are", "does", "do", "can", "has", "have", "will", "was", "were"}:
        return "yesno"
    if any(re.match(p, q) for p in YESNO_PATTERNS):
        return "yesno"
    return "vqa"


def is_korean(text: str) -> bool:
    """한글 음절 (Hangul Syllables) 1개 이상 포함 시 True."""
    return any(0xAC00 <= ord(c) <= 0xD7A3 for c in text)


# ──────────────────────────────────────────────────────────────────
# Enhanced wrapper
# ──────────────────────────────────────────────────────────────────
class EnhancedVLM:
    """v3 추론 + 5개 inference-time 기법 통합."""

    def __init__(
        self,
        model: MiniLLaVA,
        ood_detector: OODDetector,
        ood_threshold: float = 0.55,
        enable_translation: bool = True,
        enable_back_translation: bool = False,  # NEW: Helsinki en-ko 가 망가져 있어 default False
        enable_clip_subject: bool = True,
        device: Optional[str] = None,
        pope_threshold: float = 0.0,  # 0.015 (POPE bench) → 0.0 (demo 친화적)
    ):
        self.model = model
        self.ood = ood_detector
        self.ood_threshold = ood_threshold
        self.pope_threshold = pope_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.enable_translation = enable_translation  # KO→EN translation
        self.enable_back_translation = enable_back_translation  # EN→KO translation (broken)
        self.enable_clip_subject = enable_clip_subject

        # MT 모델 lazy load
        self._mt_ko_en = None
        self._mt_en_ko = None
        self._mt_loaded = False

        # CLIP 보조 분류용 텍스트 임베딩 사전 캐시
        self._clip_class_emb = None
        if enable_clip_subject:
            self._build_clip_class_embeddings()

    def _build_clip_class_embeddings(self):
        """OODDetector 의 CLIP 으로 카테고리 임베딩 사전 계산."""
        with torch.no_grad():
            text_inputs = self.ood.processor(
                text=CLIP_CLASSES_PROMPTS, return_tensors="pt", padding=True
            ).to(self.ood.device)
            text_result = self.ood.clip.get_text_features(**text_inputs)
            text_emb = text_result.pooler_output if hasattr(text_result, "pooler_output") else text_result
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        self._clip_class_emb = text_emb

    def _ensure_mt_loaded(self, need_back: bool = False):
        if self._mt_loaded:
            return
        from transformers import MarianMTModel, MarianTokenizer
        print("[enhanced] loading MT KO→EN (~300 MB) ...")
        self._mt_ko_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
        self._mt_ko_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").to(self.device).eval()
        if need_back:
            print("[enhanced] loading MT EN→KO (~300 MB) ...")
            self._mt_en_ko_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ko")
            self._mt_en_ko = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ko").to(self.device).eval()
        self._mt_loaded = True
        print("[enhanced] MT loaded.")

    @torch.no_grad()
    def _translate(self, text: str, model, tokenizer) -> str:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def translate_ko_to_en(self, text: str) -> str:
        self._ensure_mt_loaded(need_back=False)
        return self._translate(text, self._mt_ko_en, self._mt_ko_en_tok)

    def translate_en_to_ko(self, text: str) -> str:
        self._ensure_mt_loaded(need_back=True)
        return self._translate(text, self._mt_en_ko, self._mt_en_ko_tok)

    @torch.no_grad()
    def _clip_subject(self, image: Image.Image) -> tuple[str, float]:
        """CLIP 으로 이미지의 main subject 분류 → (label, similarity)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_inputs = self.ood.processor(images=image, return_tensors="pt").to(self.ood.device)
        img_result = self.ood.clip.get_image_features(**img_inputs)
        img_emb = img_result.pooler_output if hasattr(img_result, "pooler_output") else img_result
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ self._clip_class_emb.T).squeeze(0)
        idx = int(sims.argmax().item())
        return CLIP_LABEL[idx], float(sims[idx].item())

    @torch.no_grad()
    def _clip_yesno_grounding(self, image: Image.Image, object_name: str) -> tuple[str, float]:
        """POPE 스타일 'is there X?' 질문에 CLIP 으로 직접 yes/no 응답.

        두 텍스트 prompt 의 similarity 차이로 결정:
          A: "a photo with {object}"
          B: "a photo without {object}"
        sim(A) > sim(B) + margin → yes, 아니면 no.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        prompts = [
            f"a photo containing a {object_name}",
            f"a photo with a {object_name} clearly visible",
            f"a photo without any {object_name}",
            f"a photo that does not contain a {object_name}",
        ]
        text_inputs = self.ood.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self.ood.device)
        text_result = self.ood.clip.get_text_features(**text_inputs)
        text_emb = text_result.pooler_output if hasattr(text_result, "pooler_output") else text_result
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        img_inputs = self.ood.processor(images=image, return_tensors="pt").to(self.ood.device)
        img_result = self.ood.clip.get_image_features(**img_inputs)
        img_emb = img_result.pooler_output if hasattr(img_result, "pooler_output") else img_result
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb @ text_emb.T).squeeze(0)  # [4]
        # mean(yes-prompts) vs mean(no-prompts)
        yes_score = (sims[0].item() + sims[1].item()) / 2
        no_score = (sims[2].item() + sims[3].item()) / 2
        margin = yes_score - no_score
        # threshold 는 호출자가 결정 — pope_threshold attr (default +0.015) 사용
        thr = getattr(self, "pope_threshold", 0.015)
        verdict = "yes" if margin > thr else "no"
        return verdict, margin

    @torch.no_grad()
    def _clip_color(self, image: Image.Image, hint: str = "object") -> tuple[str, float]:
        """이미지의 주요 색상을 CLIP zero-shot 으로 추정."""
        colors = [
            "red", "blue", "green", "yellow", "white", "black", "brown",
            "orange", "purple", "pink", "gray", "silver",
        ]
        prompts = [f"a {c} {hint}" for c in colors]
        text_inputs = self.ood.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self.ood.device)
        text_result = self.ood.clip.get_text_features(**text_inputs)
        text_emb = text_result.pooler_output if hasattr(text_result, "pooler_output") else text_result
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        img_inputs = self.ood.processor(images=image, return_tensors="pt").to(self.ood.device)
        img_result = self.ood.clip.get_image_features(**img_inputs)
        img_emb = img_result.pooler_output if hasattr(img_result, "pooler_output") else img_result
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb @ text_emb.T).squeeze(0)
        idx = int(sims.argmax().item())
        return colors[idx], float(sims[idx].item())

    def _generate_raw(
        self,
        image: Image.Image,
        question: str,
        max_new: int,
        repetition_penalty: float = 1.0,
    ) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixel_values = self.model.image_processor(image, return_tensors="pt")["pixel_values"].to(self.model.llm.device)
        input_ids, attn = encode_for_inference(self.model.tokenizer, question)
        input_ids = input_ids.unsqueeze(0).to(self.model.llm.device)
        attn = attn.unsqueeze(0).to(self.model.llm.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=pixel_values,
                max_new_tokens=max_new,
                do_sample=False,  # greedy → deterministic
            )
        return self.model.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def answer(self, image: Image.Image, question: str, return_meta: bool = False):
        """메인 entry — 모든 enhancement 적용."""
        meta = {}

        # 1. 언어 감지
        ko = is_korean(question)
        meta["lang"] = "ko" if ko else "en"

        # 2. 한국어면 영문으로 번역
        if ko and self.enable_translation:
            try:
                en_question = self.translate_ko_to_en(question)
                meta["translated_question"] = en_question
            except Exception as e:
                print(f"[enhanced] KO→EN MT failed: {e}")
                en_question = question
                meta["translated_question"] = None
        else:
            en_question = question

        # 3. 질문 type 분류 — Korean 이고 번역 안 됐으면 korean qtype 강제
        if ko and not (self.enable_translation and meta.get("translated_question")):
            qtype = "korean"
        else:
            qtype = detect_question_type(en_question)
        cfg = QTYPE_CONFIG.get(qtype, QTYPE_CONFIG["default"])
        meta["qtype"] = qtype
        meta["max_new"] = cfg["max_new"]

        # 4. ★★ POPE-style 'is there X?' → CLIP grounding 으로 직접 yes/no
        # (v3/v2 모두 무조건 'Yes' bias 가 있으므로 CLIP 으로 우회)
        m = re.match(
            r"\s*(?:is|are)\s+there\s+(?:an?|the|any)\s+(.+?)\s+(?:in|on|at)\s+the\s+(?:image|picture|photo)",
            en_question.lower(),
        )
        if m and self.enable_clip_subject:
            obj = m.group(1).strip().strip("?,. ")
            verdict, margin = self._clip_yesno_grounding(image, obj)
            meta["clip_grounding_obj"] = obj
            meta["clip_grounding_margin"] = margin
            meta["clip_grounding_verdict"] = verdict
            meta["raw_answer_en"] = f"[clip-grounded] {verdict} (margin {margin:+.4f})"
            extracted = verdict
            # 한국어 처리
            if ko and self.enable_translation:
                final = "네." if verdict == "yes" else "아니요."
            else:
                final = verdict
            meta["final"] = final
            meta["used_path"] = "clip_grounding_yesno"
            return (final, meta) if return_meta else final

        # 5. raw 생성 (위 분기 안 탔을 때)
        raw = self._generate_raw(image, en_question, max_new=cfg["max_new"])
        meta["raw_answer_en"] = raw

        # 6. OOD 평가 (gate)
        try:
            clip_sim, clip_match = self.ood.clip_similarity(image)
            meta["clip_sim"] = clip_sim
            meta["clip_match"] = clip_match
            is_ood_simple = clip_sim < 0.20  # 임계값
            meta["is_ood"] = is_ood_simple
        except Exception as e:
            print(f"[enhanced] OOD eval failed: {e}")
            is_ood_simple = False
            meta["is_ood"] = None

        # 7. ★★ 색상 질문 → CLIP color zero-shot
        # 매칭 패턴: "what color", "which color", "what's the color", "what is the color", "the color of"
        _q_lower = en_question.lower()
        is_color_q = (
            "color" in _q_lower
            and (
                _q_lower.startswith(("what color", "which color"))
                or _q_lower.startswith(("what's the color", "what is the color"))
                or "color of" in _q_lower
            )
        )
        if self.enable_clip_subject and is_color_q:
            color, color_conf = self._clip_color(image)
            meta["clip_color"] = color
            meta["clip_color_conf"] = color_conf
            extracted = color
            if ko and self.enable_translation:
                color_ko_map = {
                    "red": "빨간색", "blue": "파란색", "green": "초록색",
                    "yellow": "노란색", "white": "흰색", "black": "검은색",
                    "brown": "갈색", "orange": "주황색", "purple": "보라색",
                    "pink": "분홍색", "gray": "회색", "silver": "은색",
                }
                final = color_ko_map.get(color, color)
            else:
                final = color
            meta["final"] = final
            meta["used_path"] = "clip_color"
            return (final, meta) if return_meta else final

        # 8. Output extraction (qtype 별)
        if qtype == "yesno":
            extracted = extract_yesno(raw)
        elif qtype in {"vqa", "color", "count"}:
            extracted = extract_short_answer(raw)
        else:
            extracted = raw

        # 9. ★★ "what is" + 단답 → CLIP subject 분류 cross-check
        if (
            self.enable_clip_subject
            and qtype == "vqa"
            and en_question.lower().startswith(("what is in", "what's in", "what is this", "what's this"))
        ):
            clip_label, clip_conf = self._clip_subject(image)
            meta["clip_subject_label"] = clip_label
            meta["clip_subject_conf"] = clip_conf
            # CLIP 이 confident 하고 v3 응답에 포함 안 되면 override
            if clip_conf > 0.22 and clip_label.lower() not in extracted.lower():
                meta["clip_override"] = True
                extracted = clip_label
            else:
                meta["clip_override"] = False

        # 10. OOD 게이트 — describe 외 질문에서 OOD 시 abstain
        if is_ood_simple and qtype not in {"yesno"} and meta.get("used_path") != "clip_color":
            extracted = "I don't know" if not ko else "잘 모르겠습니다"
            meta["ood_gated"] = True
        else:
            meta["ood_gated"] = False

        # 11. 한국어 사용자 응답 처리
        #     - back_translation 비활성 (default): 영어 답변에 "[영어 답변]" prefix
        #       (Helsinki opus-mt-tc-big-en-ko 가 gibberish 생성 → 정확한 영어가 더 나음)
        #     - back_translation 활성: KO 로 번역 시도
        if ko and not meta.get("ood_gated", False):
            if self.enable_back_translation and self.enable_translation:
                try:
                    final = self.translate_en_to_ko(extracted)
                    if not final or len(final) < 2:
                        # gibberish/empty → fallback EN
                        final = f"[영어 답변] {extracted}"
                except Exception as e:
                    print(f"[enhanced] EN→KO MT failed: {e}")
                    final = f"[영어 답변] {extracted}"
            else:
                # back-translation 비활성 → 영어 답변 + 한국어 안내
                final = f"[영어 답변] {extracted}"
        else:
            final = extracted

        meta["final"] = final
        meta["used_path"] = meta.get("used_path", "vlm_raw")
        return (final, meta) if return_meta else final
