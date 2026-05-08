"""Generic image-question-answer pipeline shared by VLM/VQA datasets.

Schema produced (per split):

- ``id``           : int64 — row index within split
- ``image``        : large_binary — inline JPEG bytes
- ``image_id``     : string?
- ``question_id``  : string?
- ``question``     : string
- ``answers``      : list<string>
- ``answer``       : string  — canonical (first) answer for FTS / quick look
- ``image_emb``    : list<float32, 512>   — CLIP image embedding (cosine-normalized)
- ``question_emb`` : list<float32, 512>   — CLIP text embedding of the question

Indices: IVF_PQ on both embeddings, FTS on ``question`` and ``answer``,
BTREE on ``image_id`` / ``question_id`` when present.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional

import pyarrow as pa
from PIL import Image

import lance

from .embeddings import CLIPEncoder
from .indexing import build_default_indices
from .schemas import fixed_size_emb_field


MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def build_schema(emb_dim: int, *, extra_fields: Optional[List[pa.Field]] = None) -> pa.Schema:
    fields: List[pa.Field] = [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("image", pa.large_binary(), nullable=False),
        pa.field("image_id", pa.string(), nullable=True),
        pa.field("question_id", pa.string(), nullable=True),
        pa.field("question", pa.string(), nullable=False),
        pa.field("answers", pa.list_(pa.string()), nullable=False),
        pa.field("answer", pa.string(), nullable=False),
        fixed_size_emb_field("image_emb", emb_dim),
        fixed_size_emb_field("question_emb", emb_dim),
    ]
    if extra_fields:
        fields.extend(extra_fields)
    return pa.schema(fields)


def _encode_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def write_split(
    *,
    rows_iter: Iterator[dict],
    n_rows: int,
    out_path: Path,
    encoder: CLIPEncoder,
    extra_fields: Optional[List[pa.Field]] = None,
    extra_value_fn: Optional[Callable[[List[dict]], dict]] = None,
    batch_size: int = 128,
    overwrite: bool = False,
) -> Path:
    """Iterate over ``rows_iter`` (each dict: image, question, answers, ...)
    and write a Lance dataset under ``out_path``.

    ``extra_fields`` + ``extra_value_fn`` let datasets attach additional
    columns (e.g. ``ocr_tokens`` for TextVQA, ``docId`` for DocVQA).
    """

    out_path = Path(out_path)
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = build_schema(encoder.DIM, extra_fields=extra_fields)
    state = {"written": 0}

    def batches() -> Iterator[pa.RecordBatch]:
        cur: List[dict] = []
        for row in rows_iter:
            cur.append(row)
            if len(cur) >= batch_size:
                yield _flush(cur, schema, encoder, extra_value_fn, state["written"])
                state["written"] += len(cur)
                if state["written"] % max(batch_size * 4, 4096) < batch_size:
                    print(f"  {state['written']:,}/{n_rows:,} rows", flush=True)
                cur = []
        if cur:
            yield _flush(cur, schema, encoder, extra_value_fn, state["written"])
            state["written"] += len(cur)
            print(f"  {state['written']:,}/{n_rows:,} rows", flush=True)

    print(f"Writing {n_rows:,} rows -> {out_path}", flush=True)
    lance.write_dataset(
        batches(),
        str(out_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )
    return out_path


def _flush(
    rows: List[dict],
    schema: pa.Schema,
    encoder: CLIPEncoder,
    extra_value_fn: Optional[Callable[[List[dict]], dict]],
    base_id: int,
) -> pa.RecordBatch:
    pil = [r["image"] for r in rows]
    image_bytes = [_encode_jpeg(im) for im in pil]
    questions = [str(r.get("question") or "") for r in rows]
    answers_list = [list(r.get("answers") or []) for r in rows]
    canonical = [(a[0] if a else (str(r.get("answer") or ""))) for a, r in zip(answers_list, rows)]

    img_emb = encoder.encode_images(pil, batch_size=min(128, len(rows)))
    q_emb = encoder.encode_texts(questions, batch_size=min(512, len(rows)))

    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "image_id": [(str(r["image_id"]) if r.get("image_id") is not None else None) for r in rows],
        "question_id": [(str(r["question_id"]) if r.get("question_id") is not None else None) for r in rows],
        "question": questions,
        "answers": answers_list,
        "answer": canonical,
        "image_emb": img_emb.tolist(),
        "question_emb": q_emb.tolist(),
    }

    if extra_value_fn is not None:
        data.update(extra_value_fn(rows))

    return pa.RecordBatch.from_pydict(data, schema=schema)


def index_split(
    out_path: Path,
    *,
    extra_btree: Iterable[str] = (),
    extra_bitmap: Iterable[str] = (),
    extra_label_list: Iterable[str] = (),
) -> None:
    ds = lance.dataset(str(out_path))
    btree_cols = ["image_id", "question_id", *extra_btree]
    btree_cols = [c for c in btree_cols if c in ds.schema.names]
    build_default_indices(
        ds,
        vector_columns=("image_emb", "question_emb"),
        fts_columns=("question", "answer"),
        btree_columns=tuple(btree_cols),
        bitmap_columns=tuple(extra_bitmap),
        label_list_columns=tuple(extra_label_list),
        metric="cosine",
    )
