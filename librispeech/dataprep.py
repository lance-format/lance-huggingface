#!/usr/bin/env python3
"""Convert LibriSpeech (openslr/librispeech_asr, ``clean`` config) to Lance.

Bundles the canonical clean evaluation splits and ``train.clean.100``:

| Split (HF)   | Lance file              | Rows   | ~ GB |
|--------------|-------------------------|--------|------|
| train.100    | train_clean_100.lance   | 28,539 | 6.6 |
| validation   | dev_clean.lance         |  2,703 | 0.36 |
| test         | test_clean.lance        |  2,620 | 0.37 |

Each row is one utterance. The original FLAC bytes are stored inline (no
re-encoding); the transcript is embedded with sentence-transformers
``all-MiniLM-L6-v2`` (384-d, cosine-normalized) so semantic search across
transcripts works out of the box.

Schema:
- ``id`` : string — LibriSpeech utterance id (e.g. ``1272-128104-0000``)
- ``audio`` : large_binary — inline FLAC bytes (16 kHz mono)
- ``sampling_rate`` : int32 — always 16,000
- ``text`` : string — transcript
- ``speaker_id`` : int64
- ``chapter_id`` : int64
- ``num_chars`` : int32 — len(text)
- ``text_emb`` : fixed_size_list<float32, 384>
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


HF_REPO_ID = "lance-format/librispeech-clean-lance"
SOURCE_REPO = "openslr/librispeech_asr"
SOURCE_CONFIG = "clean"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024

# (HF split name, output filename without extension)
SPLITS = (
    ("validation", "dev_clean"),
    ("test",       "test_clean"),
    ("train.100",  "train_clean_100"),
)


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("audio", pa.large_binary(), nullable=False),
            pa.field("sampling_rate", pa.int32(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("speaker_id", pa.int64(), nullable=False),
            pa.field("chapter_id", pa.int64(), nullable=False),
            pa.field("num_chars", pa.int32(), nullable=False),
            fixed_size_emb_field("text_emb", emb_dim),
        ]
    )


def _flush(rows: List[dict], encoder: SentenceEncoder, schema: pa.Schema) -> pa.RecordBatch:
    texts = [r["text"] for r in rows]
    emb = encoder.encode_texts(texts)
    data = {
        "id": [r["id"] for r in rows],
        "audio": [r["audio"]["bytes"] for r in rows],
        "sampling_rate": [16000 for _ in rows],
        "text": texts,
        "speaker_id": [int(r["speaker_id"]) for r in rows],
        "chapter_id": [int(r["chapter_id"]) for r in rows],
        "num_chars": [len(r["text"]) for r in rows],
        "text_emb": emb.tolist(),
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
        vector_columns=("text_emb",),
        fts_columns=("text",),
        btree_columns=("id", "speaker_id", "chapter_id"),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="LibriSpeech clean -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "librispeech-clean-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    args = p.parse_args()

    from datasets import load_dataset, Audio

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = SentenceEncoder()

    for hf_split_name, out_name in SPLITS:
        hf = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=hf_split_name)
        # Skip decoding so we keep the raw FLAC bytes from the source parquet.
        hf = hf.cast_column("audio", Audio(sampling_rate=16000, decode=False))
        out_split = data_root / f"{out_name}.lance"
        write_split(hf, out_split, encoder, batch_size=512, overwrite=args.overwrite)
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
