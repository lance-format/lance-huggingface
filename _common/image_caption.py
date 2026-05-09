"""Build a Lance dataset from an HF image-caption dataset.

Schema produced (per split):

- ``id``: int64 — row index within the split
- ``image``: large_binary — raw image bytes (JPEG)
- ``image_id``: string — original image id from the source
- ``filename``: string?
- ``captions``: list<string> — every caption attached to the image
- ``caption``: string — the canonical (first) caption for FTS / quick browsing
- ``image_emb``: list<float32, dim>? — CLIP image embedding
- ``text_emb``: list<float32, dim>? — CLIP text embedding of the canonical caption

Indices created:
- IVF_PQ on ``image_emb``
- IVF_PQ on ``text_emb``
- INVERTED on ``caption``
- BTREE on ``image_id``
"""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pyarrow as pa
from PIL import Image

import lance

from .embeddings import CLIPEncoder
from .indexing import build_default_indices
from .schemas import fixed_size_emb_field


MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def build_schema(*, embed: bool, emb_dim: int) -> pa.Schema:
    fields = [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("image", pa.large_binary(), nullable=False),
        pa.field("image_id", pa.string(), nullable=True),
        pa.field("filename", pa.string(), nullable=True),
        pa.field("captions", pa.list_(pa.string()), nullable=False),
        pa.field("caption", pa.string(), nullable=False),
    ]
    if embed:
        fields.append(fixed_size_emb_field("image_emb", emb_dim))
        fields.append(fixed_size_emb_field("text_emb", emb_dim))
    return pa.schema(fields)


def _encode_jpeg(img: Image.Image, quality: int = 92) -> bytes:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def write_split(
    *,
    rows_iter,
    n_rows: int,
    out_path: Path,
    encoder: Optional[CLIPEncoder] = None,
    batch_size: int = 256,
    overwrite: bool = False,
) -> Path:
    """Write a Lance dataset given an iterable of dicts with keys
    {image: PIL, captions: List[str], image_id: str?, filename: str?}.
    """
    out_path = Path(out_path)
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(f"{out_path} exists; pass overwrite=True")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embed = encoder is not None
    schema = build_schema(embed=embed, emb_dim=encoder.DIM if encoder else 0)

    print(f"Writing {n_rows:,} rows -> {out_path}", flush=True)

    state = {"written": 0}

    def batches() -> Iterator[pa.RecordBatch]:
        cur: List[dict] = []
        for row in rows_iter:
            cur.append(row)
            if len(cur) >= batch_size:
                yield _flush(cur, schema, encoder, embed, state["written"])
                state["written"] += len(cur)
                if state["written"] % max(batch_size * 4, 4096) < batch_size:
                    print(f"  {state['written']:,}/{n_rows:,} rows", flush=True)
                cur = []
        if cur:
            yield _flush(cur, schema, encoder, embed, state["written"])
            state["written"] += len(cur)
            print(f"  {state['written']:,}/{n_rows:,} rows", flush=True)

    lance.write_dataset(
        batches(),
        str(out_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )
    return out_path


def _flush(
    rows: Sequence[dict],
    schema: pa.Schema,
    encoder: Optional[CLIPEncoder],
    embed: bool,
    base_id: int,
) -> pa.RecordBatch:
    pil_images: List[Image.Image] = [r["image"] for r in rows]
    image_bytes = [_encode_jpeg(im) for im in pil_images]
    captions_list = [list(r.get("captions") or [r.get("caption", "")]) for r in rows]
    canonical = [(c[0] if c else "") for c in captions_list]

    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "image_id": [str(r.get("image_id")) if r.get("image_id") is not None else None for r in rows],
        "filename": [r.get("filename") for r in rows],
        "captions": captions_list,
        "caption": canonical,
    }
    if embed:
        data["image_emb"] = encoder.encode_images(pil_images, batch_size=min(256, len(rows))).tolist()
        data["text_emb"] = encoder.encode_texts(canonical, batch_size=min(1024, len(rows))).tolist()

    return pa.RecordBatch.from_pydict(data, schema=schema)


def index_split(out_path: Path, *, has_emb: bool) -> None:
    ds = lance.dataset(str(out_path))
    build_default_indices(
        ds,
        vector_columns=("image_emb", "text_emb") if has_emb else (),
        fts_columns=("caption",),
        btree_columns=("image_id",),
        metric="cosine",
    )
