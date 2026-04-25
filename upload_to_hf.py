# Setup:
#   pip install huggingface_hub
#   huggingface-cli login
#
# Run:
#   python upload_to_hf.py

import os
import sys
import traceback
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Rohithreddybc/judgesense-benchmark"
REPO_TYPE = "dataset"
DATASET_URL = f"https://huggingface.co/datasets/{REPO_ID}"
COMMIT_MESSAGE = "Upload judgesense-benchmark dataset"

IGNORE_PATTERNS = [
    "**/.git*",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/.env",
    "**/*.log",
]

SCRIPT_DIR = Path(__file__).parent.resolve()
FOLDER_PATH = SCRIPT_DIR / "judgesense-benchmark"


def list_upload_files(folder: Path) -> list[Path]:
    skip_parts = {".git", "__pycache__"}
    files = []
    for f in sorted(folder.rglob("*")):
        if not f.is_file():
            continue
        if any(part in skip_parts or part.startswith(".") for part in f.parts):
            continue
        if f.suffix in {".pyc", ".pyo", ".log"}:
            continue
        files.append(f)
    return files


def main():
    if not FOLDER_PATH.exists():
        print(f"ERROR: Folder not found: {FOLDER_PATH}", file=sys.stderr)
        sys.exit(1)

    files = list_upload_files(FOLDER_PATH)
    if not files:
        print("ERROR: No files found to upload.", file=sys.stderr)
        sys.exit(1)

    print(f"Files to upload ({len(files)}):")
    for f in files:
        print(f"  {f.relative_to(FOLDER_PATH)}")
    print()

    api = HfApi()

    try:
        api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
        print(f"Uploading to {REPO_ID} ...")

        api.upload_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(FOLDER_PATH),
            commit_message=COMMIT_MESSAGE,
            ignore_patterns=IGNORE_PATTERNS,
        )

        print(f"\nSUCCESS: Dataset uploaded.")
        print(f"View at: {DATASET_URL}")

    except Exception:
        traceback.print_exc()
        print("\nFAILURE: Upload did not complete. See error above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
