"""v3 Step 2 — VRAM + step time 사전 측정.

ViT-L/14 (576 patches, 1024 hidden) + LoRA Stage 2 가 8GB VRAM 에 들어가는지,
1 forward+backward 가 얼마나 걸리는지 측정해 학습 scope 결정에 사용.

사용 (33 디렉토리에서):
  python scripts/vram_test_step2.py
  python scripts/vram_test_step2.py --batch-size 1 --grad-accum 8
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import VISION_MODEL_L14  # noqa: E402
from src.dataset import VQACollator, VQADataset  # noqa: E402
from src.model import MiniLLaVA  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="실제 학습에선 사용. test 자체에선 step 1번만 측정.")
    p.add_argument("--data-path", type=str,
                   default="data/v3_step1_korean/manifest.json")
    p.add_argument("--max-text-length", type=int, default=512)
    p.add_argument("--n-steps", type=int, default=3,
                   help="warmup 1 step + 측정 N-1 step (정확한 step time 평균)")
    return p.parse_args()


def gb(bytes_):
    return bytes_ / (1024 ** 3)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 사용 불가")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()

    print("=" * 70)
    print(f"v3 Step 2 — VRAM + step time 측정")
    print(f"  vision  : {VISION_MODEL_L14}")
    print(f"  dtype   : bfloat16")
    print(f"  bsz     : {args.batch_size} (grad_accum={args.grad_accum} → effective={args.batch_size*args.grad_accum})")
    print("=" * 70)

    # ─────── [1/4] 모델 로드 ───────
    t0 = time.time()
    print(f"\n[1/4] MiniLLaVA + ViT-L/14 로드 (bf16) ...")
    model = MiniLLaVA(
        vision_model_name=VISION_MODEL_L14,
        freeze_vision=True,
        freeze_llm=False,  # PEFT 가 base 를 freeze
        torch_dtype=torch.bfloat16,
    )
    print(f"   load 완료 ({time.time()-t0:.1f}s)")

    # LoRA wrap
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)

    print(f"   trainable params: {model.num_trainable():,}")

    print(f"\n[2/4] GPU 이동 ...")
    model.to(device)
    after_load_gb = gb(torch.cuda.memory_allocated())
    print(f"   after load → memory_allocated = {after_load_gb:.2f} GB")

    # ─────── [2/4] 데이터 로드 ───────
    print(f"\n[3/4] 데이터 1 batch 로드 ({args.data_path}) ...")
    if not Path(args.data_path).exists():
        print(f"[FAIL] manifest 없음: {args.data_path}")
        return
    dataset = VQADataset(
        args.data_path, model.tokenizer, model.image_processor,
        max_length=args.max_text_length,
    )
    collator = VQACollator(pad_token_id=model.tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, collate_fn=collator)

    optimizer = AdamW(model.trainable_parameters(), lr=2e-4)

    # ─────── [3/4] forward+backward 측정 (warmup + repeat) ───────
    print(f"\n[4/4] {args.n_steps} step 측정 (1 warmup + {args.n_steps-1} 측정) ...")

    step_times = []
    for i, batch in enumerate(loader):
        if i >= args.n_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_time = time.time() - t0
        peak = gb(torch.cuda.max_memory_allocated())
        cur = gb(torch.cuda.memory_allocated())
        tag = "[warmup]" if i == 0 else "[measure]"
        print(f"   {tag} step {i+1}: loss={loss.item():.4f}  "
              f"time={step_time:.2f}s  cur={cur:.2f}GB  peak={peak:.2f}GB")
        if i > 0:
            step_times.append(step_time)

    avg_step = sum(step_times) / max(1, len(step_times))
    peak_final = gb(torch.cuda.max_memory_allocated())

    # ─────── 결과 분석 ───────
    print()
    print("=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"  평균 step time   : {avg_step:.2f}s ({args.batch_size} samples × forward+backward+opt)")
    print(f"  peak VRAM        : {peak_final:.2f} / 8.0 GB ({100*peak_final/8:.1f}%)")
    print()

    if peak_final > 7.8:
        print("  🔴 OOM 위험 — gradient checkpointing 또는 batch=1 이하 필요")
        print("     권장: model.gradient_checkpointing_enable() 추가 + 재측정")
    elif peak_final > 7.0:
        print("  ⚠️  빡빡 — batch=1 가능하나 다른 process 가 GPU 잡으면 OOM")
        print("     권장: 학습 중 다른 GPU process 차단 + monitoring")
    else:
        print(f"  ✅ 안전 — {8-peak_final:.1f}GB 여유. batch={args.batch_size} grad_accum={args.grad_accum} 학습 가능")
        print()
        print("  학습 시간 추정 (effective batch = {}):".format(args.batch_size*args.grad_accum))
        # 가정: dataset N samples → N/effective_batch optimizer steps
        # step_time (1 forward) × grad_accum = optimizer step time
        opt_step_time = avg_step * args.grad_accum
        for n_samples, name in [(5_000, "Stage 1 (Flickr30k)"),
                                (13_000, "Stage 2 (mix data)")]:
            n_opt = n_samples // (args.batch_size * args.grad_accum)
            for n_epoch in [1, 2]:
                total_min = n_opt * n_epoch * opt_step_time / 60
                print(f"    {name}: {n_samples} samples × {n_epoch}ep "
                      f"= {n_opt*n_epoch} opt steps × {opt_step_time:.1f}s "
                      f"≈ {total_min:.0f}분 ({total_min/60:.1f}h)")

    print("=" * 70)


if __name__ == "__main__":
    main()
