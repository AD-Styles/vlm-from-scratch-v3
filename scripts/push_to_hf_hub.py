"""HF Hub 에 v3 가중치 push.

업로드 대상 (~14 MB total):
  projector.pt                            5.7 MB
  lora_adapter_slim/adapter_config.json   1 KB
  lora_adapter_slim/adapter_model.safetensors  8.27 MB
  lora_adapter_slim/image_token_row.safetensors  7 KB
  lora_adapter_slim/README.md             5 KB (PEFT auto)
  README.md                               (custom model card, hf_hub_README.md → README.md)

저장소: AD-Styles/mini-llava-v3
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

REPO_ID = "AD-Styles/mini-llava-v3"
LOCAL_CKPT = Path("checkpoints/v3_step1_korean")
PROJECTOR = LOCAL_CKPT / "projector.pt"
SLIM_ADAPTER = LOCAL_CKPT / "lora_adapter_slim"
MODEL_CARD = Path("hf_hub_README.md")


def main():
    api = HfApi()
    me = api.whoami()
    print(f"[hub] logged in as: {me['name']}")

    # 1. repo 생성 (이미 있으면 skip)
    print(f"\n[hub] creating repo {REPO_ID} (if not exists) ...")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"[hub] repo ready: https://huggingface.co/{REPO_ID}")

    # 2. README (model card) 업로드
    if not MODEL_CARD.exists():
        raise FileNotFoundError(f"model card 없음: {MODEL_CARD}")
    print(f"\n[hub] uploading model card: {MODEL_CARD} → README.md")
    upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add v3 model card (Korean + Slim + OOD)",
    )
    print("[hub] README.md uploaded")

    # 3. projector.pt 업로드
    if not PROJECTOR.exists():
        raise FileNotFoundError(f"projector 없음: {PROJECTOR}")
    size_mb = PROJECTOR.stat().st_size / 1024 / 1024
    print(f"\n[hub] uploading projector ({size_mb:.2f} MB): {PROJECTOR}")
    upload_file(
        path_or_fileobj=str(PROJECTOR),
        path_in_repo="projector.pt",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add v3 projector (Korean training, 175 min)",
    )
    print("[hub] projector.pt uploaded")

    # 4. slim adapter 폴더 통째로 업로드
    if not SLIM_ADAPTER.exists():
        raise FileNotFoundError(f"slim adapter 없음: {SLIM_ADAPTER}")
    total_mb = sum(p.stat().st_size for p in SLIM_ADAPTER.glob("*")) / 1024 / 1024
    print(f"\n[hub] uploading slim adapter ({total_mb:.2f} MB): {SLIM_ADAPTER}")
    upload_folder(
        folder_path=str(SLIM_ADAPTER),
        path_in_repo="lora_adapter_slim",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add v3 slim adapter (1045MB → 8.28MB, no quality loss)",
    )
    print("[hub] lora_adapter_slim/ uploaded")

    # 5. 검증 — repo 의 파일 목록 확인
    print(f"\n[hub] verifying repo contents ...")
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    print(f"[hub] repo files ({len(files)}):")
    for f in sorted(files):
        print(f"  {f}")

    print(f"\n[OK] v3 weights uploaded -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
