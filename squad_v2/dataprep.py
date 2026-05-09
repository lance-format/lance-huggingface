#!/usr/bin/env python3
"""Convert SQuAD v2 (rajpurkar/squad_v2) to Lance with text embeddings.

Schema:
- ``id``           : string row id from SQuAD
- ``title``        : Wikipedia article title
- ``context``      : passage paragraph
- ``question``     : the question
- ``answers``      : list<string> — accepted answers (empty for impossible questions)
- ``answer_starts``: list<int32>  — char offsets within ``context``
- ``is_impossible``: bool
- ``question_emb`` : 384-d MiniLM embedding of the question (cosine-normalized)

Indices:
- IVF_PQ on ``question_emb``
- INVERTED on ``context`` and ``question``
- BTREE on ``id`` and ``title``
- BITMAP on ``is_impossible``
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pyarrow as pa

import lance

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import SentenceEncoder
from _common.indexing import build_default_indices
from _common.schemas import fixed_size_emb_field
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/squad-v2-lance"
SOURCE_REPO = "rajpurkar/squad_v2"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("context", pa.string(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
            pa.field("answers", pa.list_(pa.string()), nullable=False),
            pa.field("answer_starts", pa.list_(pa.int32()), nullable=False),
            pa.field("is_impossible", pa.bool_(), nullable=False),
            fixed_size_emb_field("question_emb", emb_dim),
        ]
    )


def _to_batches(hf_split, encoder: SentenceEncoder, batch_size: int) -> Iterator[pa.RecordBatch]:
    schema = _build_schema(encoder.DIM)
    cur: List[dict] = []
    n = len(hf_split)
    for i, row in enumerate(hf_split):
        cur.append(row)
        if len(cur) >= batch_size:
            yield _flush(cur, encoder, schema)
            print(f"  {i + 1:,}/{n:,} rows", flush=True)
            cur = []
    if cur:
        yield _flush(cur, encoder, schema)
        print(f"  {n:,}/{n:,} rows", flush=True)


def _flush(rows: List[dict], encoder: SentenceEncoder, schema: pa.Schema) -> pa.RecordBatch:
    questions = [r["question"] for r in rows]
    emb = encoder.encode_texts(questions)
    answers = [list(r["answers"]["text"]) for r in rows]
    starts = [[int(s) for s in r["answers"]["answer_start"]] for r in rows]
    data = {
        "id": [r["id"] for r in rows],
        "title": [r["title"] for r in rows],
        "context": [r["context"] for r in rows],
        "question": questions,
        "answers": answers,
        "answer_starts": starts,
        "is_impossible": [len(a) == 0 for a in answers],
        "question_emb": emb.tolist(),
    }
    return pa.RecordBatch.from_pydict(data, schema=schema)


def write_split(hf_split, out_path: Path, encoder: SentenceEncoder, batch_size: int, overwrite: bool) -> None:
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _build_schema(encoder.DIM)
    n = len(hf_split)
    print(f"Writing {n:,} rows -> {out_path}", flush=True)
    lance.write_dataset(
        _to_batches(hf_split, encoder, batch_size),
        str(out_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )


def index_split(out_path: Path) -> None:
    ds = lance.dataset(str(out_path))
    build_default_indices(
        ds,
        vector_columns=("question_emb",),
        fts_columns=("question", "context"),
        btree_columns=("id", "title"),
        bitmap_columns=("is_impossible",),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="SQuAD v2 -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "squad-v2-lance"))
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

    if args.no_embed:
        raise SystemExit("--no-embed unsupported for squad_v2 (schema requires question_emb)")
    encoder = SentenceEncoder()

    for split, out_name in (("train", "train"), ("validation", "validation")):
        hf = load_dataset(SOURCE_REPO, split=split)
        out_split = data_root / f"{out_name}.lance"
        write_split(hf, out_split, encoder, batch_size=2048, overwrite=args.overwrite)
        if not args.no_index:
            index_split(out_split)

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    if card.exists():
        (out_root / "README.md").write_text(card.read_text())

    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
