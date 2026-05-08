#!/usr/bin/env python3
"""Convert Natural Questions validation (google-research-datasets/natural_questions) to Lance.

Each NQ row is one Google search query with a full Wikipedia article and 1-5
annotator labels. The validation split (~7,800 rows / 3.5 GB) is bundled by
default; the train split is **143 GB** and is intentionally deferred.

Schema (per row):
- ``id`` : string — NQ example id
- ``question`` : string — search query
- ``document_title`` : string
- ``document_url`` : string
- ``document_html`` : large_binary — full HTML of the Wikipedia article (inline)
- ``short_answers`` : list<string> — every short-answer span across all annotators (deduped)
- ``num_short_answers`` : int32 — total span count
- ``has_short_answer`` : bool — at least one annotator gave a short-answer span
- ``has_long_answer`` : bool — at least one annotator selected a long-answer candidate
- ``yes_no_answer`` : string — ``YES`` / ``NO`` / ``NONE`` (most-common over annotators)
- ``question_emb`` : fixed_size_list<float32, 384>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
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


HF_REPO_ID = "lance-format/natural-questions-val-lance"
SOURCE_REPO = "google-research-datasets/natural_questions"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
            pa.field("document_title", pa.string(), nullable=False),
            pa.field("document_url", pa.string(), nullable=False),
            pa.field("document_html", pa.large_binary(), nullable=False),
            pa.field("short_answers", pa.list_(pa.string()), nullable=False),
            pa.field("num_short_answers", pa.int32(), nullable=False),
            pa.field("has_short_answer", pa.bool_(), nullable=False),
            pa.field("has_long_answer", pa.bool_(), nullable=False),
            pa.field("yes_no_answer", pa.string(), nullable=False),
            fixed_size_emb_field("question_emb", emb_dim),
        ]
    )


_YESNO = {-1: "NONE", 0: "NO", 1: "YES"}


def _flush(rows: List[dict], encoder: SentenceEncoder, schema: pa.Schema) -> pa.RecordBatch:
    questions = [r["question"]["text"] if isinstance(r["question"], dict) else str(r["question"]) for r in rows]
    emb = encoder.encode_texts(questions)
    titles, urls, htmls = [], [], []
    short_answers_per_row, num_short, has_short, has_long, yesno = [], [], [], [], []
    for r in rows:
        doc = r.get("document") or {}
        titles.append(str(doc.get("title") or ""))
        urls.append(str(doc.get("url") or ""))
        html = doc.get("html") or ""
        htmls.append(html.encode("utf-8") if isinstance(html, str) else (html or b""))

        ann = r.get("annotations") or {}
        sa = ann.get("short_answers") or []
        spans: List[str] = []
        for s in sa:
            for t in (s.get("text") or []):
                if t:
                    spans.append(str(t))
        # Dedupe while preserving order.
        seen = set()
        uniq = [t for t in spans if not (t in seen or seen.add(t))]
        short_answers_per_row.append(uniq)
        num_short.append(len(spans))
        has_short.append(len(spans) > 0)

        la = ann.get("long_answer") or []
        any_long = False
        for entry in la:
            ci = entry.get("candidate_index", -1) if isinstance(entry, dict) else -1
            if ci is not None and ci != -1:
                any_long = True
                break
        has_long.append(any_long)

        yn_list = ann.get("yes_no_answer") or []
        yn_codes = [v for v in yn_list if v != -1]
        if yn_codes:
            most = Counter(yn_codes).most_common(1)[0][0]
            yesno.append(_YESNO.get(most, "NONE"))
        else:
            yesno.append("NONE")

    data = {
        "id": [str(r["id"]) for r in rows],
        "question": questions,
        "document_title": titles,
        "document_url": urls,
        "document_html": htmls,
        "short_answers": short_answers_per_row,
        "num_short_answers": num_short,
        "has_short_answer": has_short,
        "has_long_answer": has_long,
        "yes_no_answer": yesno,
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
        fts_columns=("question",),
        btree_columns=("id", "document_title"),
        bitmap_columns=("yes_no_answer", "has_short_answer", "has_long_answer"),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Natural Questions validation -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "natural-questions-val-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["validation"])
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = SentenceEncoder()

    for split in args.splits:
        hf = load_dataset(SOURCE_REPO, split=split)
        out_split = data_root / f"{split}.lance"
        write_split(hf, out_split, encoder, batch_size=128, overwrite=args.overwrite)
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
