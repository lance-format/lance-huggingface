#!/usr/bin/env python3
"""Convert MS MARCO v2.1 (microsoft/ms_marco) to Lance.

Each MS MARCO row is a query with up to 10 candidate passages and human-written
answers. We keep one row per query and store the passages as parallel
list-typed columns (Lance handles list types natively, so this remains a
single tabular dataset). For IR-style training, users can flatten passages
client-side; for RAG / answer evaluation the per-query layout is more useful.

Schema:
- ``query_id``               : int64
- ``query``                  : string
- ``query_type``             : string  (DESCRIPTION/NUMERIC/ENTITY/...)
- ``answers``                : list<string>
- ``well_formed_answers``    : list<string>
- ``passage_text``           : list<string>
- ``passage_url``            : list<string>
- ``passage_is_selected``    : list<int8>  — 1 if relevant
- ``selected_passage``       : string?  — first selected passage, joined into a single column for FTS
- ``query_emb``              : 384-d MiniLM embedding of the query
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import pyarrow as pa
import lance

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import SentenceEncoder
from _common.indexing import build_default_indices
from _common.schemas import fixed_size_emb_field
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/ms-marco-v2.1-lance"
SOURCE_REPO = "microsoft/ms_marco"
SOURCE_CONFIG = "v2.1"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("query_id", pa.int64(), nullable=False),
            pa.field("query", pa.string(), nullable=False),
            pa.field("query_type", pa.string(), nullable=True),
            pa.field("answers", pa.list_(pa.string()), nullable=False),
            pa.field("well_formed_answers", pa.list_(pa.string()), nullable=False),
            pa.field("passage_text", pa.list_(pa.string()), nullable=False),
            pa.field("passage_url", pa.list_(pa.string()), nullable=False),
            pa.field("passage_is_selected", pa.list_(pa.int8()), nullable=False),
            pa.field("selected_passage", pa.string(), nullable=True),
            fixed_size_emb_field("query_emb", emb_dim),
        ]
    )


def _flush(rows: List[dict], encoder: SentenceEncoder, schema: pa.Schema) -> pa.RecordBatch:
    queries = [r["query"] for r in rows]
    emb = encoder.encode_texts(queries)
    selected: List[Optional[str]] = []
    for r in rows:
        passages = list(r["passages"]["passage_text"])
        sel = list(r["passages"]["is_selected"])
        first_idx = next((i for i, s in enumerate(sel) if s), None)
        selected.append(passages[first_idx] if first_idx is not None else None)
    data = {
        "query_id": [int(r["query_id"]) for r in rows],
        "query": queries,
        "query_type": [r.get("query_type") for r in rows],
        "answers": [list(r.get("answers") or []) for r in rows],
        "well_formed_answers": [list(r.get("wellFormedAnswers") or []) for r in rows],
        "passage_text": [list(r["passages"]["passage_text"]) for r in rows],
        "passage_url": [list(r["passages"]["url"]) for r in rows],
        "passage_is_selected": [
            [int(s) for s in r["passages"]["is_selected"]] for r in rows
        ],
        "selected_passage": selected,
        "query_emb": emb.tolist(),
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
        vector_columns=("query_emb",),
        fts_columns=("query", "selected_passage"),
        btree_columns=("query_id",),
        bitmap_columns=("query_type",),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="MS MARCO v2.1 -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "ms-marco-v2.1-lance"))
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
