#!/usr/bin/env python3
"""Convert ChartQA (lmms-lab/ChartQA) to Lance.

VQA over scientific / business charts. The lmms-lab redistribution exposes
``test`` only (2,500 rows) — train + val live elsewhere; extend the script
to point at additional sources if needed.
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


HF_REPO_ID = "lance-format/chartqa-lance"
SOURCE_REPO = "lmms-lab/ChartQA"


def _row_iter(hf_split):
    for r in hf_split:
        ans = r.get("answer")
        answers = ans if isinstance(ans, list) else [str(ans or "")]
        yield {
            "image": r["image"],
            "image_id": None,
            "question_id": None,
            "question": r.get("question") or "",
            "answers": answers,
            "type": r.get("type"),
        }


def _extra_fields() -> List[pa.Field]:
    return [pa.field("type", pa.string(), nullable=True)]


def _extra_values(rows):
    return {"type": [r.get("type") for r in rows]}


def main() -> None:
    p = argparse.ArgumentParser(description="ChartQA -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "chartqa-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["test"])
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
            index_split(out_split, extra_bitmap=("type",))

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())
    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
