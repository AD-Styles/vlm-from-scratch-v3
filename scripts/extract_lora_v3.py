"""v3 의 진짜 slim LoRA adapter 추출 — 1GB → ~8MB (99.2% 절감, 품질 손실 0).

배경 — v2 의 `scripts/extract_lora.py` 가 실패한 이유:
  v2 는 embed_tokens / lm_head 를 *통째로* drop → `<image>` 토큰의 학습된
  representation 까지 손실 → 응답 품질 명확히 저하 (Dog→Cat, White→Brown 등).

v3 의 깨달음 (Phase 0.1.5 사전 검증으로 확정):
  saved embed_tokens 의 첫 151665 행 = base Qwen2.5 와 정확히 일치 (max diff 0.0).
  실제로 학습된 부분은 **마지막 1 행 (`<image>` 토큰) 뿐**.
  → LoRA 8MB + `<image>` 1 행 (~7KB) 만 보존하면 정확히 동일한 모델 재구성 가능.

이 스크립트의 동작:
  1. 원본 adapter_model.safetensors → LoRA 키만 (8MB) → 새 adapter_model.safetensors
  2. embed_tokens / lm_head 의 마지막 1 행 → image_token_row.safetensors (~7KB)
  3. adapter_config.json 의 modules_to_save 항목 제거
  → 추론 시 PEFT 로 LoRA 로드 + image_token_row 를 수동 패치

사용:
  python scripts/extract_lora_v3.py \
    --input-dir checkpoints/v3_step1_korean/lora_adapter \
    --output-dir checkpoints/v3_step1_korean/lora_adapter_slim
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True,
                   help="원본 PEFT adapter 디렉터리")
    p.add_argument("--output-dir", type=str, required=True,
                   help="slim adapter 저장 위치")
    p.add_argument("--dry-run", action="store_true",
                   help="저장 없이 분석만")
    return p.parse_args()


def fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.2f} MB"


def fmt_kb(num_bytes: int) -> str:
    return f"{num_bytes / 1024:.2f} KB"


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    src_safetensors = in_dir / "adapter_model.safetensors"
    src_config = in_dir / "adapter_config.json"
    if not src_safetensors.exists():
        raise FileNotFoundError(f"원본 adapter 없음: {src_safetensors}")
    if not src_config.exists():
        raise FileNotFoundError(f"adapter_config.json 없음: {src_config}")

    # ── 1. Load original
    print(f"[load] {src_safetensors}")
    state = load_file(str(src_safetensors))
    total_in = sum(v.numel() * v.element_size() for v in state.values())
    print(f"[stat] 원본: {len(state)} 키, {fmt_mb(total_in)}")

    # ── 2. Separate LoRA / special modules
    lora_state, special_state = {}, {}
    for k, v in state.items():
        if "lora_" in k:
            lora_state[k] = v
        elif "embed_tokens" in k or "lm_head" in k:
            special_state[k] = v
        else:
            # 다른 키 (e.g., 파라미터 외 metadata 같은 것이 있을 수 있음)
            lora_state[k] = v
            print(f"  [warn] 분류 불가 키, LoRA 쪽으로 보존: {k}")

    print(f"[split] LoRA 키: {len(lora_state)} ({fmt_mb(sum(v.numel()*v.element_size() for v in lora_state.values()))})")
    print(f"[split] special 키: {len(special_state)} ({fmt_mb(sum(v.numel()*v.element_size() for v in special_state.values()))})")

    # ── 3. Extract last row (image token) from special modules
    image_rows = {}
    for k, v in special_state.items():
        last_row = v[-1].clone()  # (hidden,)
        # 키 이름 정규화: 마지막 모듈 이름 사용 (embed_tokens or lm_head)
        if "embed_tokens" in k:
            new_key = "image_token.embed_tokens"
        elif "lm_head" in k:
            new_key = "image_token.lm_head"
        else:
            print(f"  [warn] unknown special key: {k}, skipping")
            continue
        image_rows[new_key] = last_row
        print(f"  [extract] {k}[-1] (shape={tuple(v.shape)}) -> {new_key} (shape={tuple(last_row.shape)})")

    if args.dry_run:
        print("\n[dry-run] 저장 없이 종료.")
        return

    # ── 4. Save slim files
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_lora = out_dir / "adapter_model.safetensors"
    save_file(lora_state, str(dst_lora))
    print(f"\n[save] {dst_lora} — {fmt_mb(dst_lora.stat().st_size)}")

    dst_image_row = out_dir / "image_token_row.safetensors"
    save_file(image_rows, str(dst_image_row))
    print(f"[save] {dst_image_row} — {fmt_kb(dst_image_row.stat().st_size)}")

    # ── 5. Modify adapter_config.json
    with open(src_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    original_mts = config.get("modules_to_save")
    if original_mts:
        print(f"[config] modules_to_save 제거: {original_mts}")
        config["modules_to_save"] = None
    dst_config = out_dir / "adapter_config.json"
    with open(dst_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[save] {dst_config}")

    # README.md 복사
    src_readme = in_dir / "README.md"
    if src_readme.exists():
        shutil.copy(src_readme, out_dir / "README.md")

    # ── 6. Final stats
    total_out = dst_lora.stat().st_size + dst_image_row.stat().st_size
    print(f"\n{'='*60}")
    print(f"[done] slim adapter 추출 완료 → {out_dir}")
    print(f"{'='*60}")
    print(f"  원본 adapter:   {fmt_mb(total_in):>12}")
    print(f"  slim LoRA:      {fmt_mb(dst_lora.stat().st_size):>12}")
    print(f"  image row:      {fmt_kb(dst_image_row.stat().st_size):>12}")
    print(f"  slim total:     {fmt_mb(total_out):>12}")
    print(f"  reduction:      {(1 - total_out / total_in) * 100:>10.2f}%")
    print(f"{'='*60}")
    print()
    print("[다음 검증] slim adapter 로 추론:")
    print("  src/model.py 의 load_lora_adapter 가 image_token_row.safetensors 를")
    print("  자동 감지해 마지막 row 패치합니다 (이번 commit 에서 추가됨).")


if __name__ == "__main__":
    main()
