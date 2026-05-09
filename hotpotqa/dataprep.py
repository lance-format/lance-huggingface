#!/usr/bin/env python3
"""Convert HotpotQA distractor (hotpot_qa) to Lance with MiniLM embeddings.

HotpotQA is the canonical multi-hop QA benchmark. Each question requires
combining facts from multiple Wikipedia paragraphs. The ``distractor``
config gives 10 candidate paragraphs per question (gold + distractors).

Schema:
- ``id`` : string — HotpotQA question id
- ``question`` : string
- ``answer`` : string — short answer (yes / no / span)
- ``type`` : string — ``bridge`` or ``comparison``
- ``level`` : string — ``easy`` / ``medium`` / ``hard``
- ``supporting_titles`` : list<string> — Wikipedia titles that contain the gold facts
- ``supporting_sent_ids`` : list<int32> — sentence indices within those titles
- ``context_titles`` : list<string> — all 10 paragraph titles
- ``context_sentences`` : list<list<string>> — sentences per paragraph
- ``context_text`` : string — flattened paragraphs (used for FTS)
- ``num_supporting_facts`` : int32
- ``question_emb`` : 384-d MiniLM embedding (cosine-normalized)
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


HF_REPO_ID = "lance-format/hotpotqa-distractor-lance"
SOURCE_REPO = "hotpot_qa"
SOURCE_CONFIG = "distractor"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
            pa.field("answer", pa.string(), nullable=False),
            pa.field("type", pa.string(), nullable=True),
            pa.field("level", pa.string(), nullable=True),
            pa.field("supporting_titles", pa.list_(pa.string()), nullable=False),
            pa.field("supporting_sent_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("context_titles", pa.list_(pa.string()), nullable=False),
            pa.field("context_sentences", pa.list_(pa.list_(pa.string())), nullable=False),
            pa.field("context_text", pa.string(), nullable=False),
            pa.field("num_supporting_facts", pa.int32(), nullable=False),
            fixed_size_emb_field("question_emb", emb_dim),
        ]
    )


def _flush(rows: List[dict], encoder: SentenceEncoder, schema: pa.Schema) -> pa.RecordBatch:
    questions = [r["question"] for r in rows]
    emb = encoder.encode_texts(questions)
    sup_titles = []
    sup_sent_ids = []
    ctx_titles = []
    ctx_sents = []
    ctx_text_flat = []
    num_sup = []
    for r in rows:
        sf = r.get("supporting_facts") or {}
        st = list(sf.get("title") or [])
        ss = [int(x) for x in (sf.get("sent_id") or [])]
        sup_titles.append(st)
        sup_sent_ids.append(ss)
        num_sup.append(len(st))

        ctx = r.get("context") or {}
        titles = list(ctx.get("title") or [])
        sentences = [list(s) for s in (ctx.get("sentences") or [])]
        ctx_titles.append(titles)
        ctx_sents.append(sentences)
        # Flatten for FTS.
        flat = []
        for t, sents in zip(titles, sentences):
            flat.append(f"{t}: " + " ".join(sents))
        ctx_text_flat.append("\n\n".join(flat))

    data = {
        "id": [str(r["id"]) for r in rows],
        "question": questions,
        "answer": [str(r.get("answer") or "") for r in rows],
        "type": [r.get("type") for r in rows],
        "level": [r.get("level") for r in rows],
        "supporting_titles": sup_titles,
        "supporting_sent_ids": sup_sent_ids,
        "context_titles": ctx_titles,
        "context_sentences": ctx_sents,
        "context_text": ctx_text_flat,
        "num_supporting_facts": num_sup,
        "question_emb": emb.tolist(),
    }
    return pa.RecordBatch.from_pydict(data, schema=schema)


def _to_batches(hf_split, encoder: SentenceEncoder, batch_size: int) -> Iterator[pa.RecordBatch]:
    schema = _build_schema(encoder.DIM)
    cur: List[dict] = []
    n = len(hf_split)
    written = 0
    for i, row in enumerate(hf_split):
        cur.append(row)
        if len(cur) >= batch_size:
            yield _flush(cur, encoder, schema)
            written += len(cur)
            print(f"  {written:,}/{n:,} rows", flush=True)
            cur = []
    if cur:
        yield _flush(cur, encoder, schema)
        written += len(cur)
        print(f"  {written:,}/{n:,} rows", flush=True)


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
        fts_columns=("question", "context_text"),
        btree_columns=("id", "answer"),
        bitmap_columns=("type", "level"),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="HotpotQA distractor -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "hotpotqa-distractor-lance"))
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
