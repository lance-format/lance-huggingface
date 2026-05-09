#!/usr/bin/env python3
"""Convert ADE20K (1aurent/ADE20K) to Lance.

ADE20K is the canonical scene-parsing benchmark with ~3,000 object categories.
The 1aurent mirror exposes the full annotation richness (multiple segmentation
maps + instance maps + polygon-level object metadata). We keep only the bits
that are most useful out-of-the-box and store everything inline:

Schema:
- ``id`` : int64 — row index within split
- ``image`` : large_binary — inline JPEG bytes
- ``segmentation`` : large_binary — first semantic segmentation map (PNG bytes, RGB-encoded as in ADE20K)
- ``instance`` : large_binary | null — first instance map (PNG bytes), null if absent
- ``filename`` : string — ADE20K relative filename
- ``scene`` : list<string> — scene class labels (e.g. ``["bathroom"]``)
- ``object_names`` : list<string> — full per-object name list (one entry per polygon, not deduped)
- ``objects_present`` : list<string> — deduped object names — feeds the ``LABEL_LIST`` index
- ``num_objects`` : int32
- ``image_emb`` : fixed_size_list<float32, 512> — OpenCLIP ViT-B/32 image embedding (cosine-normalized)

Indices: IVF_PQ on ``image_emb``, BTREE on ``num_objects``,
``LABEL_LIST`` on ``objects_present``.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import pyarrow as pa
from PIL import Image

import lance

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.embeddings import CLIPEncoder
from _common.indexing import build_default_indices
from _common.schemas import fixed_size_emb_field
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/ade20k-lance"
SOURCE_REPO = "1aurent/ADE20K"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024


def _build_schema(emb_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("segmentation", pa.large_binary(), nullable=False),
            pa.field("instance", pa.large_binary(), nullable=True),
            pa.field("filename", pa.string(), nullable=False),
            pa.field("scene", pa.list_(pa.string()), nullable=False),
            pa.field("object_names", pa.list_(pa.string()), nullable=False),
            pa.field("objects_present", pa.list_(pa.string()), nullable=False),
            pa.field("num_objects", pa.int32(), nullable=False),
            fixed_size_emb_field("image_emb", emb_dim),
        ]
    )


def _encode(img: Image.Image, fmt: str) -> bytes:
    if fmt == "JPEG":
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "L", "RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=92, optimize=False)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _flush(rows: List[dict], encoder: CLIPEncoder, schema: pa.Schema, base_id: int) -> pa.RecordBatch:
    pil_images = [r["image"] for r in rows]
    image_bytes = [_encode(im, "JPEG") for im in pil_images]

    seg_bytes: List[bytes] = []
    inst_bytes: List[Optional[bytes]] = []
    for r in rows:
        seg_list = r.get("segmentations") or []
        inst_list = r.get("instances") or []
        seg_bytes.append(_encode(seg_list[0], "PNG") if seg_list else b"")
        inst_bytes.append(_encode(inst_list[0], "PNG") if inst_list else None)

    object_names_per_row: List[List[str]] = []
    objects_present_per_row: List[List[str]] = []
    num_objects_per_row: List[int] = []
    for r in rows:
        names = [(o.get("name") or "") for o in (r.get("objects") or [])]
        names = [n for n in names if n]
        object_names_per_row.append(names)
        objects_present_per_row.append(sorted(set(names)))
        num_objects_per_row.append(len(names))

    emb = encoder.encode_images(pil_images, batch_size=min(64, len(rows)))

    data = {
        "id": [base_id + i for i in range(len(rows))],
        "image": image_bytes,
        "segmentation": seg_bytes,
        "instance": inst_bytes,
        "filename": [r.get("filename") or "" for r in rows],
        "scene": [list(r.get("scene") or []) for r in rows],
        "object_names": object_names_per_row,
        "objects_present": objects_present_per_row,
        "num_objects": num_objects_per_row,
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
        label_list_columns=("objects_present",),
        metric="cosine",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="ADE20K -> Lance")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "lance_cache" / "ade20k-lance"))
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

    for split, out_name in (("validation", "validation"), ("train", "train")):
        hf = load_dataset(SOURCE_REPO, split=split)
        out_split = data_root / f"{out_name}.lance"
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
