"""HF Spaces 에 v3 demo 배포.

배포 구조 (Space repo root):
  app.py             ← space/app.py
  README.md          ← space/README.md (HF Spaces metadata 포함)
  requirements.txt   ← space/requirements.txt
  src/               ← src/ 전체 (model.py, ood_detection.py, dataset.py, config.py, infer.py, __init__.py)
  assets/            ← assets/ (예시 이미지 + README)

저장소: AD-Styles/mini-llava-v3-demo
가중치는 첫 실행 시 AD-Styles/mini-llava-v3 에서 자동 다운로드 (~14 MB).
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

REPO_ID = "AD-Styles/mini-llava-v3-demo"
ROOT = Path(__file__).resolve().parent.parent

# (local_path, path_in_repo) 매핑
ROOT_FILES = [
    (ROOT / "space" / "app.py", "app.py"),
    (ROOT / "space" / "README.md", "README.md"),
    (ROOT / "space" / "requirements.txt", "requirements.txt"),
]

FOLDERS = [
    (ROOT / "src", "src"),
    (ROOT / "assets", "assets"),
]

# src/ 에서 제외 (배포 불필요)
SRC_IGNORE = ["__pycache__", "*.pyc", "*.pyo"]


def main():
    api = HfApi()
    me = api.whoami()
    print(f"[spaces] logged in as: {me['name']}")

    # 1. repo 생성 (Gradio SDK)
    print(f"\n[spaces] ensuring repo {REPO_ID} (Gradio SDK) ...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    print(f"[spaces] repo ready: https://huggingface.co/spaces/{REPO_ID}")

    # 2. root 파일 업로드 (app.py, README.md, requirements.txt)
    print(f"\n[spaces] uploading root files ...")
    for local, in_repo in ROOT_FILES:
        if not local.exists():
            raise FileNotFoundError(f"missing: {local}")
        size_kb = local.stat().st_size / 1024
        print(f"  - {in_repo} ({size_kb:.1f} KB)")
        upload_file(
            path_or_fileobj=str(local),
            path_in_repo=in_repo,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message=f"Deploy v3: {in_repo}",
        )

    # 3. 폴더 업로드 (src/, assets/)
    for local, in_repo in FOLDERS:
        if not local.exists():
            raise FileNotFoundError(f"missing folder: {local}")
        files = [p for p in local.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        total_kb = sum(p.stat().st_size for p in files) / 1024
        print(f"\n[spaces] uploading {in_repo}/ ({len(files)} files, {total_kb:.1f} KB)")
        for p in files:
            print(f"  - {p.relative_to(local)}")
        upload_folder(
            folder_path=str(local),
            path_in_repo=in_repo,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message=f"Deploy v3: {in_repo}/",
            ignore_patterns=SRC_IGNORE,
        )

    # 4. 검증
    print(f"\n[spaces] verifying repo contents ...")
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="space")
    print(f"[spaces] repo files ({len(files)}):")
    for f in sorted(files):
        print(f"  {f}")

    print(f"\n[OK] v3 demo deployed -> https://huggingface.co/spaces/{REPO_ID}")
    print("    (Space build takes 3-5 minutes on first push.)")


if __name__ == "__main__":
    main()
