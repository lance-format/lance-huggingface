#!/usr/bin/env python3
"""Convert Doctor's Handwritten Prescription BD images to Lance.

The source provides cropped handwritten medication words and their labels. This
conversion preserves the original PNG bytes and creates the text/search fields
used by the companion OCR retrieval project, without generating embeddings.

Indices per split:
- FTS on ``searchable_summary``
- BTREE on ``id``
- BITMAP on ``category`` and ``needs_human_review``
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Iterator

import lance
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _common.indexing import build_default_indices
from _common.upload import push_to_hub


HF_REPO_ID = "lance-format/handwriting-ocr"
SOURCE_DATASET = "mamun1113/doctors-handwritten-prescription-bd-dataset"
MAX_BYTES_PER_FILE = 8 * 1024 * 1024 * 1024

SPLITS = {
    "train": ("Training", "training_labels.csv", "training_words"),
    "validation": ("Validation", "validation_labels.csv", "validation_words"),
    "test": ("Testing", "testing_labels.csv", "testing_words"),
}


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("medicine_name", pa.string(), nullable=False),
            pa.field("generic_name", pa.string(), nullable=False),
            pa.field("normalized_text", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("is_medical", pa.bool_(), nullable=False),
            pa.field("needs_human_review", pa.bool_(), nullable=False),
            pa.field("searchable_summary", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("image_filename", pa.string(), nullable=False),
        ]
    )


def _rows(source_root: Path, lance_split: str) -> Iterator[dict]:
    source_split, labels_name, words_name = SPLITS[lance_split]
    split_root = source_root / source_split
    labels_path = split_root / labels_name
    words_dir = split_root / words_name

    with labels_path.open(newline="", encoding="utf-8") as labels_file:
        for row_number, row in enumerate(csv.DictReader(labels_file)):
            image_filename = row["IMAGE"]
            medicine_name = row["MEDICINE_NAME"].strip()
            generic_name = row["GENERIC_NAME"].strip()
            with (words_dir / image_filename).open("rb") as image_file:
                image = image_file.read()

            yield {
                "id": f"{lance_split}_{row_number:05d}",
                "image": image,
                "medicine_name": medicine_name,
                "generic_name": generic_name,
                "normalized_text": medicine_name,
                "category": "medication",
                "is_medical": True,
                "needs_human_review": False,
                "searchable_summary": f"Medication mention: {medicine_name} ({generic_name})",
                "source_dataset": SOURCE_DATASET,
                "split": source_split,
                "image_filename": image_filename,
            }


def _batches(source_root: Path, lance_split: str, batch_size: int) -> Iterator[pa.RecordBatch]:
    schema = _schema()
    batch: list[dict] = []
    for row in _rows(source_root, lance_split):
        batch.append(row)
        if len(batch) >= batch_size:
            yield pa.RecordBatch.from_pylist(batch, schema=schema)
            batch = []
    if batch:
        yield pa.RecordBatch.from_pylist(batch, schema=schema)


def write_split(source_root: Path, lance_split: str, out_path: Path, *, batch_size: int, overwrite: bool) -> None:
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {lance_split} -> {out_path}")
    lance.write_dataset(
        _batches(source_root, lance_split, batch_size),
        str(out_path),
        schema=_schema(),
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )


def index_split(out_path: Path) -> None:
    dataset = lance.dataset(str(out_path))
    build_default_indices(
        dataset,
        fts_columns=("searchable_summary",),
        btree_columns=("id",),
        bitmap_columns=("category", "needs_human_review"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Doctor's Handwritten Prescription BD -> Lance")
    parser.add_argument(
        "--source-data",
        default=str(REPO_ROOT.parent / "ocr-handwriting-retrieval" / "data"),
        help="Path containing the Training, Validation, and Testing directories.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT.parent / "lance_cache" / "handwriting-ocr"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    args = parser.parse_args()

    source_root = Path(args.source_data)
    out_root = Path(args.out)
    for source_split, labels_name, words_name in SPLITS.values():
        for path in (source_root / source_split / labels_name, source_root / source_split / words_name):
            if not path.exists():
                raise FileNotFoundError(path)

    for lance_split in SPLITS:
        out_split = out_root / "data" / f"{lance_split}.lance"
        write_split(source_root, lance_split, out_split, batch_size=args.batch_size, overwrite=args.overwrite)
        if not args.no_index:
            index_split(out_split)

    card = Path(__file__).parent / "HF_DATASET_CARD.md"
    (out_root / "README.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")

    if args.push:
        url = push_to_hub(repo_id=args.repo_id, folder_path=out_root)
        print(f"Done: {url}")


if __name__ == "__main__":
    main()
