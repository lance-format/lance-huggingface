#!/usr/bin/env python3
"""Convert TextVQA (lmms-lab/textvqa) to Lance.

Visual question answering where the question requires reading text in the
image (street signs, product labels, screen captures). Per-row OCR tokens
ride along as a list column to support OCR-aware retrieval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.upload import push_to_hub
from _common.vlm_qa import index_split, write_split


HF_REPO_ID = "lance-format/textvqa-lance"
SOURCE_REPO = "lmms-lab/textvqa"


def _row_iter(hf_split):
    for r in hf_split:
        yield {
            "image": r["image"],
            "image_id": str(r.get("image_id")),
            "question_id": str(r.get("question_id")),
            "question": r.get("question") or "",
            "answers": list(r.get("answers") or []),
            "ocr_tokens": list(r.get("ocr_tokens") or []),
            "image_classes": list(r.get("image_classes") or []),
            "set_name": r.get("set_name"),
        }


def _extra_fields() -> List[pa.Field]:
    return [
        pa.field("ocr_tokens", pa.list_(pa.string()), nullable=False),
        pa.field("image_classes", pa.list_(pa.string()), nullable=False),
        pa.field("set_name", pa.string(), nullable=True),
    ]


def _extra_values(rows):
    return {
        "ocr_tokens": [list(r.get("ocr_tokens") or []) for r in rows],
        "image_classes": [list(r.get("image_classes") or []) for r in rows],
        "set_name": [r.get("set_name") for r in rows],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TextVQA -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "textvqa-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["validation", "train"])
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    for split in args.splits:
        hf = load_dataset(SOURCE_REPO, split=split)
        n = len(hf)
        out_split = data_root / f"{split}.lance"
        write_split(
            rows_iter=_row_iter(hf),
            n_rows=n,
            out_path=out_split,
            encoder=encoder,
            extra_fields=_extra_fields(),
            extra_value_fn=_extra_values,
            batch_size=128,
            overwrite=args.overwrite,
        )
        if not args.no_index:
            index_split(out_split, extra_btree=("set_name",))

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())
    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
