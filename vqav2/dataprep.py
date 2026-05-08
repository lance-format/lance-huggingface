#!/usr/bin/env python3
"""Convert VQAv2 (lmms-lab/VQAv2) to Lance.

Each row is one (image, question, answers) triple. The image bytes ride
alongside the question, the 10 reference answers, and **two** embeddings:
- ``image_emb``: CLIP image embedding,
- ``question_emb``: CLIP *text* embedding on the question.

Both are L2-normalized, so the same IVF_PQ index can serve cross-modal
retrieval (find images for a question, find similar questions for an image).

Schema:
- ``id`` : int64
- ``image`` : large_binary (JPEG bytes)
- ``image_id`` : int64
- ``question_id`` : int64
- ``question`` : string
- ``question_type`` : string  (e.g. ``what is``, ``is the``)
- ``answer_type`` : string    (``yes/no`` | ``number`` | ``other``)
- ``multiple_choice_answer`` : string
- ``answers`` : list<string>  — 10 annotator answers
- ``answer_confidences`` : list<string>  — parallel ``yes`` / ``maybe`` / ``no``
- ``image_emb`` : fixed_size_list<float32, 512>
- ``question_emb`` : fixed_size_list<float32, 512>
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


HF_REPO_ID = "lance-format/vqav2-lance"
SOURCE_REPO = "lmms-lab/VQAv2"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("image_id", pa.int64(), nullable=False),
            pa.field("question_id", pa.int64(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
            pa.field("question_type", pa.string(), nullable=True),
            pa.field("answer_type", pa.string(), nullable=True),
            pa.field("multiple_choice_answer", pa.string(), nullable=False),
            pa.field("answers", pa.list_(pa.string()), nullable=False),
            pa.field("answer_confidences", pa.list_(pa.string()), nullable=False),
            fixed_size_emb_field("image_emb", emb_dim),
            fixed_size_emb_field("question_emb", emb_dim),
        ]
    )


def _encode_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def _flush(rows: List[dict], encoder: CLIPEncoder, schema: pa.Schema, base_id: int) -> pa.RecordBatch:
    pil = [r["image"] for r in rows]
    image_bytes = [_encode_jpeg(im) for im in pil]
    questions = [r["question"] for r in rows]

    answer_strings: List[List[str]] = []
    answer_confs: List[List[str]] = []
    for r in rows:
        a_list = r.get("answers") or []
        answer_strings.append([str(a.get("answer") or "") for a in a_list])
        answer_confs.append([str(a.get("answer_confidence") or "") for a in a_list])

    img_emb = encoder.encode_images(pil, batch_size=min(128, len(rows)))
    q_emb = encoder.encode_texts(questions, batch_size=min(512, len(rows)))

    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "image_id": [int(r["image_id"]) for r in rows],
        "question_id": [int(r["question_id"]) for r in rows],
        "question": questions,
        "question_type": [r.get("question_type") for r in rows],
        "answer_type": [r.get("answer_type") for r in rows],
        "multiple_choice_answer": [str(r.get("multiple_choice_answer") or "") for r in rows],
        "answers": answer_strings,
        "answer_confidences": answer_confs,
        "image_emb": img_emb.tolist(),
        "question_emb": q_emb.tolist(),
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
        vector_columns=("image_emb", "question_emb"),
        fts_columns=("question",),
        btree_columns=("image_id", "question_id", "multiple_choice_answer"),
        bitmap_columns=("question_type", "answer_type"),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="VQAv2 -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "vqav2-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["validation", "train"])
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

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
