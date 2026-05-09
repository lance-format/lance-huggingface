#!/usr/bin/env python3
"""Convert DocVQA (lmms-lab/DocVQA, ``DocVQA`` config) to Lance.

VQA over document images (forms, receipts, reports). Each row carries the
document page image, the question, the multiple reference answers, and the
question-type tag.
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


HF_REPO_ID = "lance-format/docvqa-lance"
SOURCE_REPO = "lmms-lab/DocVQA"
SOURCE_CONFIG = "DocVQA"


def _row_iter(hf_split):
    for r in hf_split:
        qtypes = list(r.get("question_types") or [])
        yield {
            "image": r["image"],
            "image_id": str(r.get("docId") or ""),
            "question_id": str(r.get("questionId") or ""),
            "question": r.get("question") or "",
            "answers": list(r.get("answers") or []),
            "doc_id": str(r.get("docId") or ""),
            "ucsf_document_id": r.get("ucsf_document_id"),
            "ucsf_document_page_no": r.get("ucsf_document_page_no"),
            "data_split": r.get("data_split"),
            "question_types": qtypes,
        }


def _extra_fields() -> List[pa.Field]:
    return [
        pa.field("doc_id", pa.string(), nullable=True),
        pa.field("ucsf_document_id", pa.string(), nullable=True),
        pa.field("ucsf_document_page_no", pa.string(), nullable=True),
        pa.field("data_split", pa.string(), nullable=True),
        pa.field("question_types", pa.list_(pa.string()), nullable=False),
    ]


def _extra_values(rows):
    return {
        "doc_id": [r.get("doc_id") for r in rows],
        "ucsf_document_id": [r.get("ucsf_document_id") for r in rows],
        "ucsf_document_page_no": [str(r.get("ucsf_document_page_no")) if r.get("ucsf_document_page_no") is not None else None for r in rows],
        "data_split": [r.get("data_split") for r in rows],
        "question_types": [list(r.get("question_types") or []) for r in rows],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="DocVQA -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "docvqa-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["validation", "test"])
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    for split in args.splits:
        hf = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=split)
        n = len(hf)
        out_split = data_root / f"{split}.lance"
        write_split(
            rows_iter=_row_iter(hf),
            n_rows=n,
            out_path=out_split,
            encoder=encoder,
            extra_fields=_extra_fields(),
            extra_value_fn=_extra_values,
            batch_size=64,
            overwrite=args.overwrite,
        )
        if not args.no_index:
            index_split(out_split, extra_btree=("doc_id",), extra_label_list=("question_types",))

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())
    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
