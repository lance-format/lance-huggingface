"""Helpers for pushing a Lance dataset directory to the Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


HF_REPO_TYPE = "dataset"


def push_to_hub(
    *,
    repo_id: str,
    folder_path: str | os.PathLike,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> str:
    """Create the dataset repo (if missing) and upload ``folder_path`` to it.

    Uses ``upload_large_folder`` so multi-GB Lance datasets stream up correctly.
    Returns the public URL of the dataset.
    """
    from huggingface_hub import HfApi

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"folder_path does not exist: {folder}")

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE, exist_ok=True, private=private)

    print(f"Pushing {folder} -> https://huggingface.co/datasets/{repo_id}")
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type=HF_REPO_TYPE,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_card(
    *,
    repo_id: str,
    card_path: str | os.PathLike,
    token: Optional[str] = None,
) -> None:
    """Upload a local file as ``README.md`` on the dataset repo."""
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
    )
