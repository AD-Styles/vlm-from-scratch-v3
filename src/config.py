from dataclasses import dataclass


# Vision encoder options
VISION_MODEL = "openai/clip-vit-base-patch32"
"""v1/v2 — 49 patches (7×7), 768 hidden, 224×224 input."""

VISION_MODEL_L14 = "openai/clip-vit-large-patch14-336"
"""v3 — 576 patches (24×24), 1024 hidden, 336×336 input. ~3.5× larger params,
12× longer image sequence → significantly higher VRAM / training time."""

LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "You receive an image and a question, and answer concisely and accurately."
)


@dataclass
class TrainConfig:
    data_path: str = "data/coco_subset/manifest.json"
    output_dir: str = "checkpoints/v1_baseline"

    batch_size: int = 8
    grad_accum_steps: int = 1
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_text_length: int = 512

    log_every: int = 20
    save_every: int = 500
    seed: int = 42

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Stage 2 → Stage 1 projector 이어받기용 (선택)
    init_projector: str | None = None

    # Vision encoder 선택 (v3 의 ViT-L/14 업그레이드용)
    vision_model: str = VISION_MODEL

    # tie_word_embeddings 분리 — v3 의 slim adapter 실험 (bonus)
    # True: lm_head ↔ embed_tokens 가중치 분리 → LoRA gradient 가 embed_tokens 안 건드림
    # → PEFT adapter 저장 시 embed_tokens 자동 포함 X → 1GB → ~50MB 목표
    untie_embeddings: bool = False

    # bf16 학습 — v3 의 ViT-L/14 (576 patches × 1024 hidden) 메모리 대응
    # CLIP-ViT-L/14 + Qwen2.5-0.5B + LoRA 가 8GB VRAM 에 들어가려면 bf16 필수
    bf16: bool = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
