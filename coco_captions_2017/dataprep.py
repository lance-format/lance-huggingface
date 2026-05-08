#!/usr/bin/env python3
"""Convert COCO Captions 2017 (lmms-lab/COCO-Caption2017) to Lance with CLIP embeddings.

We use ``lmms-lab/COCO-Caption2017`` because it embeds the images directly in
parquet, so the resulting Lance dataset is fully self-contained and does not
depend on cocodataset.org being reachable. This covers val2017 (5k) plus a
40.7k-row test slice.
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


HF_REPO_ID = "lance-format/coco-captions-2017-lance"
SOURCE_REPO = "lmms-lab/COCO-Caption2017"
SOURCE_SPLITS = ("val", "test")  # val=5k, test=40.7k


def _row_iter(hf_split):
    for row in hf_split:
        yield {
            "image": row["image"],
            "image_id": str(row.get("id")),
            "filename": row.get("file_name"),
            "captions": list(row.get("answer") or []),
        }


def main() -> None:
    p = argparse.ArgumentParser(description="COCO Captions 2017 -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "coco-captions-2017-lance"))
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=list(SOURCE_SPLITS))
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = None if args.no_embed else CLIPEncoder()

    for split in args.splits:
        ds = load_dataset(SOURCE_REPO, split=split)
        n_rows = len(ds)
        out_split = data_root / f"{split}.lance"
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
