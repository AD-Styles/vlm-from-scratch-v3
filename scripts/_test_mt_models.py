"""다양한 EN→KO MT 모델 빠른 테스트 — 어떤 모델이 작동하는지 확인."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

TEST_SENTENCES = [
    "In this image we can see a dog and the background is white.",
    "Yes, there is a dog in the image.",
    "The character is yellow.",
    "dog",
    "I cannot identify this image.",
]

# Candidates
CANDIDATES = [
    ("Helsinki-NLP/opus-mt-tc-big-en-ko", "marian"),
    # ("Helsinki-NLP/opus-mt-en-ko", "marian"),  # 보통 존재 안 함
    ("facebook/m2m100_418M", "m2m100"),
    ("facebook/nllb-200-distilled-600M", "nllb"),
]


def test_marian(model_name):
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(model_name)
    m = MarianMTModel.from_pretrained(model_name).eval()
    results = []
    for s in TEST_SENTENCES:
        inputs = tok(s, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=128, num_beams=4)
        results.append(tok.decode(out[0], skip_special_tokens=True))
    return results


def test_m2m100(model_name):
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    tok = M2M100Tokenizer.from_pretrained(model_name)
    m = M2M100ForConditionalGeneration.from_pretrained(model_name).eval()
    tok.src_lang = "en"
    results = []
    for s in TEST_SENTENCES:
        inputs = tok(s, return_tensors="pt")
        with torch.no_grad():
            out = m.generate(**inputs, forced_bos_token_id=tok.get_lang_id("ko"), max_new_tokens=128)
        results.append(tok.decode(out[0], skip_special_tokens=True))
    return results


def test_nllb(model_name):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    tok.src_lang = "eng_Latn"
    results = []
    for s in TEST_SENTENCES:
        inputs = tok(s, return_tensors="pt")
        with torch.no_grad():
            out = m.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids("kor_Hang"),
                max_new_tokens=128,
            )
        results.append(tok.decode(out[0], skip_special_tokens=True))
    return results


for model_name, kind in CANDIDATES:
    print(f"\n=== {model_name} ({kind}) ===")
    try:
        if kind == "marian":
            outs = test_marian(model_name)
        elif kind == "m2m100":
            outs = test_m2m100(model_name)
        elif kind == "nllb":
            outs = test_nllb(model_name)
        for s, o in zip(TEST_SENTENCES, outs):
            print(f"  EN: {s}")
            print(f"  KO: {o}")
            print()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
