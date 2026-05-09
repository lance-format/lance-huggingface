#!/usr/bin/env python3
"""Convert Fashion-MNIST (zalando-datasets/fashion_mnist) to Lance format with CLIP embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.image_classification import index_split, write_split
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/fashion-mnist-lance"
SOURCE_REPO = "zalando-datasets/fashion_mnist"


def main() -> None:
    p = argparse.ArgumentParser(description="Fashion-MNIST -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "fashion-mnist-lance"))
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

    for split in ("train", "test"):
        ds = load_dataset(SOURCE_REPO, split=split)
        class_names = [name.replace(" ", "_").replace("/", "_") for name in ds.features["label"].names]
        out_split = data_root / f"{split}.lance"
        write_split(
            hf_split=ds,
            out_path=out_split,
            class_names=class_names,
            image_col="image",
            label_col="label",
            encoder=encoder,
            encode_format="PNG",
            batch_size=2048,
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
