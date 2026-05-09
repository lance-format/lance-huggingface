"""Build a Lance dataset from an HF image-classification dataset.

Schema produced (per split):

- ``id``: int64 — row index within the split
- ``image``: large_binary — raw image bytes (re-encoded to JPEG/PNG when needed)
- ``label``: int32 — class id
- ``label_name``: string — human-readable class name
- ``image_emb``: fixed_size_list<float32, dim>? — CLIP image embedding (optional)

Indices created when present:
- IVF_PQ on ``image_emb`` (cosine, normalized)
- BITMAP on ``label_name``
- BTREE on ``label``
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pyarrow as pa
from PIL import Image

import lance

from .embeddings import CLIPEncoder
from .indexing import build_default_indices
from .schemas import fixed_size_emb_field


MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024  # 8 GiB Lance fragments


def _build_schema(class_names: List[str], *, embed: bool, emb_dim: int) -> pa.Schema:
    fields = [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("image", pa.large_binary(), nullable=False),
        pa.field("label", pa.int32(), nullable=False),
        pa.field("label_name", pa.string(), nullable=False),
    ]
    if embed:
        fields.append(fixed_size_emb_field("image_emb", emb_dim))
    schema = pa.schema(fields)
    return schema.with_metadata(
        {
            "lance:class_names": ",".join(class_names),
            "lance:source_modality": "image",
        }
    )


def _encode_image(img: Image.Image, *, fmt: str = "PNG") -> bytes:
    if fmt == "JPEG":
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format=fmt, quality=92, optimize=False)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def write_split(
    *,
    hf_split,
    out_path: Path,
    class_names: List[str],
    image_col: str,
    label_col: str,
    encoder: Optional[CLIPEncoder],
    encode_format: str = "PNG",
    batch_size: int = 1024,
    overwrite: bool = False,
    log_every: int = 5000,
) -> Path:
    out_path = Path(out_path)
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(f"{out_path} exists; use overwrite=True")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embed = encoder is not None
    emb_dim = encoder.DIM if encoder else 0
    schema = _build_schema(class_names, embed=embed, emb_dim=emb_dim)

    n = len(hf_split)
    print(f"Writing {n:,} rows -> {out_path}", flush=True)

    def batches() -> Iterator[pa.RecordBatch]:
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            chunk = hf_split.select(range(start, end))
            ids = list(range(start, end))
            pil_images = [chunk[i][image_col] for i in range(end - start)]
            labels = [int(chunk[i][label_col]) for i in range(end - start)]
            label_names = [class_names[lbl] for lbl in labels]
            image_bytes = [_encode_image(im, fmt=encode_format) for im in pil_images]

            data = {
                "id": ids,
                "image": image_bytes,
                "label": labels,
                "label_name": label_names,
            }
            if embed:
                emb = encoder.encode_images(pil_images, batch_size=min(256, batch_size))
                data["image_emb"] = emb.tolist()

            yield pa.RecordBatch.from_pydict(data, schema=schema)

            if end % max(1, log_every) < batch_size:
                print(f"  {end:,}/{n:,} rows", flush=True)

    lance.write_dataset(
        batches(),
        str(out_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )
    return out_path


def index_split(out_path: Path, *, has_emb: bool) -> None:
    ds = lance.dataset(str(out_path))
    build_default_indices(
        ds,
        vector_columns=("image_emb",) if has_emb else (),
        bitmap_columns=("label_name",),
        btree_columns=("label",),
        metric="cosine",
    )
