#!/usr/bin/env python3
"""Convert COCO 2017 object detection (detection-datasets/coco) to Lance.

Each row is one image with the full list of object annotations attached as
parallel list-typed columns. The image bytes ride alongside the annotations
inline; CLIP image embeddings are computed on the fly and stored in
``image_emb`` for visual retrieval.

Schema:
- ``id`` : int64 — row index within split
- ``image`` : large_binary — inline JPEG bytes
- ``image_id`` : int64 — COCO image id
- ``width`` : int32
- ``height`` : int32
- ``bboxes`` : list<list<float32, 4>> — ``[x_min, y_min, x_max, y_max]`` (absolute pixel coords)
- ``categories`` : list<int32> — COCO 80-class id (0-79)
- ``category_names`` : list<string>
- ``areas`` : list<float32>
- ``num_objects`` : int32
- ``categories_present`` : list<string> — deduped class names for filtering / LABEL_LIST index
- ``image_emb`` : fixed_size_list<float32, 512> — OpenCLIP ViT-B/32 image embedding (cosine-normalized)

Indices: IVF_PQ on ``image_emb``, BTREE on ``image_id`` and ``num_objects``,
LABEL_LIST on ``categories_present``.
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


HF_REPO_ID = "lance-format/coco-detection-2017-lance"
SOURCE_REPO = "detection-datasets/coco"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


# COCO 80-class ids, in dataset order (matches detection-datasets/coco label space).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("image_id", pa.int64(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("bboxes", pa.list_(pa.list_(pa.float32(), 4)), nullable=False),
            pa.field("categories", pa.list_(pa.int32()), nullable=False),
            pa.field("category_names", pa.list_(pa.string()), nullable=False),
            pa.field("areas", pa.list_(pa.float32()), nullable=False),
            pa.field("num_objects", pa.int32(), nullable=False),
            pa.field("categories_present", pa.list_(pa.string()), nullable=False),
            fixed_size_emb_field("image_emb", emb_dim),
        ]
    )


def _encode_jpeg(img: Image.Image, quality: int = 92) -> bytes:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def _flush(rows: List[dict], encoder: CLIPEncoder, schema: pa.Schema, base_id: int) -> pa.RecordBatch:
    pil = [r["image"] for r in rows]
    image_bytes = [_encode_jpeg(im) for im in pil]
    emb = encoder.encode_images(pil, batch_size=min(128, len(rows)))

    bboxes_per_row: List[List[List[float]]] = []
    cats_per_row: List[List[int]] = []
    cat_names_per_row: List[List[str]] = []
    areas_per_row: List[List[float]] = []
    num_obj_per_row: List[int] = []
    cats_present_per_row: List[List[str]] = []

    for r in rows:
        objs = r["objects"]
        boxes = [list(map(float, b)) for b in (objs.get("bbox") or [])]
        cats = [int(c) for c in (objs.get("category") or [])]
        names = [COCO_CLASSES[c] if 0 <= c < len(COCO_CLASSES) else "unknown" for c in cats]
        areas = [float(a) for a in (objs.get("area") or [])]

        bboxes_per_row.append(boxes)
        cats_per_row.append(cats)
        cat_names_per_row.append(names)
        areas_per_row.append(areas)
        num_obj_per_row.append(len(boxes))
        cats_present_per_row.append(sorted(set(names)))

    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "image_id": [int(r["image_id"]) for r in rows],
        "width": [int(r.get("width", pil[i].width)) for i, r in enumerate(rows)],
        "height": [int(r.get("height", pil[i].height)) for i, r in enumerate(rows)],
        "bboxes": bboxes_per_row,
        "categories": cats_per_row,
        "category_names": cat_names_per_row,
        "areas": areas_per_row,
        "num_objects": num_obj_per_row,
        "categories_present": cats_present_per_row,
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
        btree_columns=("image_id", "num_objects"),
        label_list_columns=("categories_present",),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="COCO Detection 2017 -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "coco-detection-2017-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["val", "train"])
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
