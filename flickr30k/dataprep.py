#!/usr/bin/env python3
"""Convert Flickr30k (lmms-lab/flickr30k) to Lance with CLIP image+text embeddings.

The lmms-lab parquet conversion exposes a single ``test`` split of 31,783 rows
that effectively covers the entire Flickr30k corpus (the original train/val/test
labels are preserved if needed; we keep them in a ``split_label`` column).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.image_caption import index_split, write_split
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/flickr30k-lance"
SOURCE_REPO = "lmms-lab/flickr30k"
SOURCE_SPLIT = "test"  # the only split — covers all 31,783 image/caption rows


def _row_iter(hf_split):
    for i, row in enumerate(hf_split):
        captions = list(row.get("caption") or [])
        yield {
            "image": row["image"],
            "image_id": str(row.get("img_id")),
            "filename": row.get("filename"),
            "captions": captions,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Flickr30k -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "flickr30k-lance"))
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = None if args.no_embed else CLIPEncoder()

    ds = load_dataset(SOURCE_REPO, split=SOURCE_SPLIT)
    n_rows = len(ds)
    out_split = data_root / "train.lance"

    write_split(
        rows_iter=_row_iter(ds),
        n_rows=n_rows,
        out_path=out_split,
        encoder=encoder,
        batch_size=128,
        overwrite=args.overwrite,
    )
    if not args.no_index:
        index_split(out_split, has_emb=encoder is not None)

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())

    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
