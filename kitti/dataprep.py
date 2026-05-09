#!/usr/bin/env python3
"""Convert KITTI 2D Object Detection (nateraw/kitti) to Lance.

KITTI is the canonical autonomous-driving 2D detection benchmark — 7,481
training images and 7,518 test images with 8 object classes. The annotations
include 2D bounding boxes plus the 3D box parameters that come from the
Velodyne stereo rig: ``alpha`` (observation angle), ``dimensions`` (h, w, l in
metres), ``location`` (3D centre in camera coords), ``rotation_y``, plus
``occluded`` and ``truncated`` flags.

Schema:
- ``id`` : int64 — row index within split
- ``image`` : large_binary — inline JPEG bytes (re-encoded from the source PNG)
- ``bboxes`` : list<list<float32, 4>> — ``[left, top, right, bottom]`` in pixel coords
- ``alphas`` : list<float32>
- ``dimensions`` : list<list<float32, 3>> — ``(h, w, l)`` in metres
- ``locations`` : list<list<float32, 3>> — 3D centre in camera coords
- ``rotation_y`` : list<float32>
- ``occluded`` : list<int8>
- ``truncated`` : list<float32>
- ``types`` : list<string> — KITTI class name per object (e.g. ``Car``, ``Pedestrian``)
- ``num_objects`` : int32
- ``types_present`` : list<string> — deduped class names — feeds the LABEL_LIST index
- ``image_emb`` : fixed_size_list<float32, 512>
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


HF_REPO_ID = "lance-format/kitti-2d-detection-lance"
SOURCE_REPO = "nateraw/kitti"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("bboxes", pa.list_(pa.list_(pa.float32(), 4)), nullable=False),
            pa.field("alphas", pa.list_(pa.float32()), nullable=False),
            pa.field("dimensions", pa.list_(pa.list_(pa.float32(), 3)), nullable=False),
            pa.field("locations", pa.list_(pa.list_(pa.float32(), 3)), nullable=False),
            pa.field("rotation_y", pa.list_(pa.float32()), nullable=False),
            pa.field("occluded", pa.list_(pa.int8()), nullable=False),
            pa.field("truncated", pa.list_(pa.float32()), nullable=False),
            pa.field("types", pa.list_(pa.string()), nullable=False),
            pa.field("num_objects", pa.int32(), nullable=False),
            pa.field("types_present", pa.list_(pa.string()), nullable=False),
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
    emb = encoder.encode_images(pil, batch_size=min(64, len(rows)))

    bboxes_per_row: List[List[List[float]]] = []
    alphas_per_row: List[List[float]] = []
    dims_per_row: List[List[List[float]]] = []
    locs_per_row: List[List[List[float]]] = []
    rot_per_row: List[List[float]] = []
    occ_per_row: List[List[int]] = []
    trunc_per_row: List[List[float]] = []
    types_per_row: List[List[str]] = []
    num_objs_per_row: List[int] = []
    types_present_per_row: List[List[str]] = []

    for r in rows:
        labels = r.get("label") or []
        boxes = [list(map(float, lab["bbox"])) for lab in labels]
        alphas = [float(lab["alpha"]) for lab in labels]
        dims = [list(map(float, lab["dimensions"])) for lab in labels]
        locs = [list(map(float, lab["location"])) for lab in labels]
        rots = [float(lab["rotation_y"]) for lab in labels]
        occ = [int(lab["occluded"]) for lab in labels]
        trunc = [float(lab["truncated"]) for lab in labels]
        types = [str(lab["type"]) for lab in labels]

        bboxes_per_row.append(boxes)
        alphas_per_row.append(alphas)
        dims_per_row.append(dims)
        locs_per_row.append(locs)
        rot_per_row.append(rots)
        occ_per_row.append(occ)
        trunc_per_row.append(trunc)
        types_per_row.append(types)
        num_objs_per_row.append(len(boxes))
        types_present_per_row.append(sorted(set(types)))

    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "bboxes": bboxes_per_row,
        "alphas": alphas_per_row,
        "dimensions": dims_per_row,
        "locations": locs_per_row,
        "rotation_y": rot_per_row,
        "occluded": occ_per_row,
        "truncated": trunc_per_row,
        "types": types_per_row,
        "num_objects": num_objs_per_row,
        "types_present": types_present_per_row,
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
        btree_columns=("num_objects",),
        label_list_columns=("types_present",),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="KITTI 2D Detection -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "kitti-2d-detection-lance"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--splits", nargs="*", default=["train"])  # test split has no labels in nateraw/kitti
    args = p.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    for split in args.splits:
        hf = load_dataset(SOURCE_REPO, split=split)
        out_split = data_root / f"{split}.lance"
        write_split(hf, out_split, encoder, batch_size=64, overwrite=args.overwrite)
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
