#!/usr/bin/env python3
"""Convert the MNIST handwritten-digit dataset (ylecun/mnist) to Lance format.

Output layout:
    out/
      data/
        train.lance
        test.lance
      README.md (uploaded separately as the dataset card)

Each row:
- ``id``      : int64 row index within split
- ``image``   : PNG bytes (28x28 grayscale)
- ``label``   : int32 (0-9)
- ``label_name`` : string ("0".."9")
- ``image_emb`` : 512-d CLIP embedding (cosine-normalized) — IVF_PQ index

Indices: IVF_PQ on ``image_emb`` + BITMAP on ``label_name`` + BTREE on ``label``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.image_classification import index_split, write_split
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/mnist-lance"
SOURCE_REPO = "ylecun/mnist"


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "mnist-lance"))
    p.add_argument("--no-embed", action="store_true", help="Skip CLIP embeddings")
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
        class_names = ds.features["label"].names
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

    # Copy the dataset card into the output dir so push_to_hub uploads it.
    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())

    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
