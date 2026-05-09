#!/usr/bin/env python3
"""Convert TriviaQA RC (mandarjoshi/trivia_qa, ``rc.nocontext`` config) to Lance.

We use the ``rc.nocontext`` config because it's the standard reading-comprehension
slice without the bulky entity_pages/search_results contexts (which would
balloon the dataset to dozens of GB).

Schema:
- ``question_id``       : string
- ``question``          : string
- ``question_source``   : string
- ``answer_value``      : string  — canonical answer
- ``answer_aliases``    : list<string>
- ``normalized_answer`` : string
- ``answer_type``       : string
- ``question_emb``      : 384-d MiniLM embedding (cosine-normalized)

Indices: IVF_PQ on ``question_emb``, INVERTED on ``question``, BTREE on
``question_id``/``answer_value``, BITMAP on ``answer_type``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator, List

import pyarrow as pa
import lance

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import SentenceEncoder
from _common.indexing import build_default_indices
from _common.schemas import fixed_size_emb_field
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/trivia-qa-lance"
SOURCE_REPO = "mandarjoshi/trivia_qa"
SOURCE_CONFIG = "rc.nocontext"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("question_id", pa.string(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
            pa.field("question_source", pa.string(), nullable=True),
            pa.field("answer_value", pa.string(), nullable=False),
            pa.field("answer_aliases", pa.list_(pa.string()), nullable=False),
            pa.field("normalized_answer", pa.string(), nullable=False),
            pa.field("answer_type", pa.string(), nullable=True),
            fixed_size_emb_field("question_emb", emb_dim),
        ]
    )


def _flush(rows: List[dict], encoder: SentenceEncoder, schema: pa.Schema) -> pa.RecordBatch:
    questions = [r["question"] for r in rows]
    emb = encoder.encode_texts(questions)
    data = {
        "question_id": [r["question_id"] for r in rows],
        "question": questions,
        "question_source": [r.get("question_source") for r in rows],
        "answer_value": [r["answer"]["value"] for r in rows],
        "answer_aliases": [list(r["answer"].get("aliases") or []) for r in rows],
        "normalized_answer": [r["answer"].get("normalized_value") or "" for r in rows],
        "answer_type": [r["answer"].get("type") for r in rows],
        "question_emb": emb.tolist(),
    }
    return pa.RecordBatch.from_pydict(data, schema=schema)


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
        fts_columns=("question",),
        btree_columns=("question_id", "answer_value"),
        bitmap_columns=("answer_type",),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="TriviaQA rc.nocontext -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "trivia-qa-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = SentenceEncoder()

    for split in ("train", "validation"):
        hf = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=split)
        out_split = data_root / f"{split}.lance"
        write_split(hf, out_split, encoder, batch_size=4096, overwrite=args.overwrite)
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
