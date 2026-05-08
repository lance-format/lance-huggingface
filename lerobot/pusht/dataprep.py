#!/usr/bin/env python3
"""Convert LeRobot's PushT (lerobot/pusht) to Lance.

This is a thin wrapper that downloads the LeRobot v3.0 dataset from the Hub
and reuses the conversion path implemented for ``lerobot/xvla-soft-fold``
(``frames.lance`` + ``videos.lance`` + ``episodes.lance``). PushT is the
canonical 2D-pushing benchmark from the Diffusion Policy paper — small enough
(~1 GB) to convert end-to-end in a few minutes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing v3.0 conversion pipeline.
sys.path.insert(0, str(REPO_ROOT / "lerobot" / "xvla-soft-fold"))
from dataprep import convert_dataset_v30_to_lance_bundle  # type: ignore


HF_REPO_ID = "lance-format/lerobot-pusht-lance"
SOURCE_REPO = "lerobot/pusht"


def _download_source(local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["data/**", "meta/**", "videos/**", "*.json"],
    )
    return local_dir


def main() -> None:
    p = argparse.ArgumentParser(description="LeRobot PushT -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "lerobot-pusht-lance"))
    p.add_argument("--source-dir", default=str(REPO_ROOT.parent / "lance_cache" / "_pusht_source"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--episode-vcodec", default="libx264")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    args = p.parse_args()

    out_root = Path(args.out)
    src_root = Path(args.source_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    src_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {SOURCE_REPO} -> {src_root}", flush=True)
    _download_source(src_root)

    bundle_root = out_root / "data"
    bundle_root.mkdir(parents=True, exist_ok=True)

    convert_dataset_v30_to_lance_bundle(
        root=src_root,
        out_root=bundle_root,
        overwrite=args.overwrite,
        limit=args.limit,
        log_every=20,
        episode_vcodec=args.episode_vcodec,
    )

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())

    if args.push:
        from _common.upload import push_to_hub  # type: ignore
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")

    if not os.environ.get("KEEP_SOURCE"):
        shutil.rmtree(src_root, ignore_errors=True)


if __name__ == "__main__":
    main()
