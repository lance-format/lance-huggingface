#!/usr/bin/env python3
"""Convert Oxford-IIIT Pet (pcuenq/oxford-pets) to Lance with CLIP embeddings.

37 cat & dog breed classes, 7,390 photos. The source has a single split, so
we publish it as ``train.lance``.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path
from typing import Iterator, List

import pyarrow as pa
from PIL import Image

import lance

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.indexing import build_default_indices
from _common.schemas import fixed_size_emb_field
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/oxford-pets-lance"
SOURCE_REPO = "pcuenq/oxford-pets"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("label_name", pa.string(), nullable=False),
            pa.field("is_dog", pa.bool_(), nullable=False),
            pa.field("path", pa.string(), nullable=True),
            fixed_size_emb_field("image_emb", emb_dim),
        ]
    )


def _encode_jpeg(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _flush(rows: List[dict], encoder: CLIPEncoder, schema: pa.Schema, base_id: int) -> pa.RecordBatch:
    pil = [r["image"] for r in rows]
    image_bytes = [_encode_jpeg(im) for im in pil]
    emb = encoder.encode_images(pil, batch_size=min(128, len(rows)))
    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "label_name": [str(r.get("label") or "").replace(" ", "_") for r in rows],
        "is_dog": [bool(r.get("dog")) for r in rows],
        "path": [r.get("path") for r in rows],
        "image_emb": emb.tolist(),
    }
    return pa.RecordBatch.from_pydict(data, schema=schema)


def _to_batches(hf_split, encoder: CLIPEncoder, batch_size: int) -> Iterator[pa.RecordBatch]:
    schema = _build_schema(encoder.DIM)
    cur: List[dict] = []
    n = len(hf_split)
    written = 0
    for i, row in enumerate(hf_split):
        cur.append(row)
        if len(cur) >= batch_size:
            yield _flush(cur, encoder, schema, written)
            written += len(cur)
            print(f"  {written:,}/{n:,} rows", flush=True)
            cur = []
    if cur:
        yield _flush(cur, encoder, schema, written)
        written += len(cur)
        print(f"  {written:,}/{n:,} rows", flush=True)


def write_split(hf_split, out_path: Path, encoder: CLIPEncoder, batch_size: int, overwrite: bool) -> None:
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
        vector_columns=("image_emb",),
        bitmap_columns=("label_name", "is_dog"),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Oxford-IIIT Pet -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "oxford-pets-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    hf = load_dataset(SOURCE_REPO, split="train")
    out_split = data_root / "train.lance"
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
