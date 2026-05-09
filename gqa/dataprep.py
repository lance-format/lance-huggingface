#!/usr/bin/env python3
"""Convert GQA testdev_balanced (lmms-lab/GQA) to Lance.

GQA is a compositional VQA benchmark with 22 question types and explicit
scene-graph reasoning programs. The lmms-lab redistribution splits it into
two parallel configs: one for instructions (Q/A), one for images. We
**join** them on ``imageId`` so each row in the resulting Lance dataset
carries the question + the matching image bytes inline.

This converter ships the canonical 12,578-question testdev-balanced split
(joined against 398 images). For the larger train_balanced or val_balanced
sets, extend the script via ``--instr-config`` / ``--images-config``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.upload import push_to_hub
from _common.vlm_qa import index_split, write_split


HF_REPO_ID = "lance-format/gqa-testdev-balanced-lance"
SOURCE_REPO = "lmms-lab/GQA"
DEFAULT_INSTR_CONFIG = "testdev_balanced_instructions"
DEFAULT_IMAGES_CONFIG = "testdev_balanced_images"
SPLIT = "testdev"


def _row_iter(instructions_split, image_lookup):
    for r in instructions_split:
        img_id = r.get("imageId")
        img = image_lookup.get(img_id)
        if img is None:
            continue
        types = r.get("types") or {}
        groups = r.get("groups") or {}
        yield {
            "image": img,
            "image_id": str(img_id),
            "question_id": str(r.get("id") or ""),
            "question": r.get("question") or "",
            "answers": [str(r.get("answer") or "")],
            "full_answer": r.get("fullAnswer"),
            "structural": types.get("structural"),
            "semantic": types.get("semantic"),
            "detailed": types.get("detailed"),
            "is_balanced": bool(r.get("isBalanced") or False),
            "group_global": groups.get("global"),
            "group_local": groups.get("local"),
            "semantic_str": r.get("semanticStr"),
        }


def _extra_fields() -> List[pa.Field]:
    return [
        pa.field("full_answer", pa.string(), nullable=True),
        pa.field("structural", pa.string(), nullable=True),
        pa.field("semantic", pa.string(), nullable=True),
        pa.field("detailed", pa.string(), nullable=True),
        pa.field("is_balanced", pa.bool_(), nullable=False),
        pa.field("group_global", pa.string(), nullable=True),
        pa.field("group_local", pa.string(), nullable=True),
        pa.field("semantic_str", pa.string(), nullable=True),
    ]


def _extra_values(rows):
    return {
        "full_answer": [r.get("full_answer") for r in rows],
        "structural": [r.get("structural") for r in rows],
        "semantic": [r.get("semantic") for r in rows],
        "detailed": [r.get("detailed") for r in rows],
        "is_balanced": [bool(r.get("is_balanced")) for r in rows],
        "group_global": [r.get("group_global") for r in rows],
        "group_local": [r.get("group_local") for r in rows],
        "semantic_str": [r.get("semantic_str") for r in rows],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GQA testdev_balanced -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "gqa-testdev-balanced-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--instr-config", default=DEFAULT_INSTR_CONFIG)
    p.add_argument("--images-config", default=DEFAULT_IMAGES_CONFIG)
    p.add_argument("--split", default=SPLIT)
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    print(f"Loading images: {args.images_config}/{args.split}", flush=True)
    images_ds = load_dataset(SOURCE_REPO, args.images_config, split=args.split)
    image_lookup = {row["id"]: row["image"] for row in images_ds}
    print(f"  loaded {len(image_lookup):,} images")

    print(f"Loading instructions: {args.instr_config}/{args.split}", flush=True)
    instr_ds = load_dataset(SOURCE_REPO, args.instr_config, split=args.split)
    n = len(instr_ds)

    out_split = data_root / f"{args.split}.lance"
    write_split(
        rows_iter=_row_iter(instr_ds, image_lookup),
        n_rows=n,
        out_path=out_split,
        encoder=encoder,
        extra_fields=_extra_fields(),
        extra_value_fn=_extra_values,
        batch_size=128,
        overwrite=args.overwrite,
    )
    if not args.no_index:
        index_split(out_split, extra_bitmap=("structural", "semantic", "detailed"))

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())
    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
