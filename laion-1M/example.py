#!/usr/bin/env python3
"""LAION-Subset Lance Dataset - Simple usage examples."""


import datasets
import lance
import pandas as pd
import pyarrow as pa

from pathlib import Path
from PIL import Image


# =============================================================================
# 1. Hugging Face streaming (metadata + captions)
# =============================================================================

def load_using_hf():
    ds = datasets.load_dataset(
        "lance-format/laion-subset",
        split="train",
        streaming=True,
    )
    print(ds)
    return ds


def get_hf_stream_batch(ds, batch_size=5):
    batch = list(ds.take(batch_size))
    df = pd.DataFrame.from_records(batch)
    print("\nHF streaming batch (pandas view):")
    print(df.head())
    return df


# =============================================================================
# 2. Lance helpers (images + ANN index)
# =============================================================================

def load_dataset():
    ds = lance.dataset("hf://datasets/lance-format/laion-subset")
    print(f"✓ Loaded {ds.count_rows():,} image-text pairs")
    return ds


def show_metadata(ds, limit=5, offset=0):
    rows = ds.scanner(
        columns=["caption", "url", "similarity", "width", "height"],
        limit=limit,
        offset=offset,
    ).to_table().to_pylist()

    print(f"\nSample metadata (rows {offset}-{offset + len(rows)}):")
    for idx, row in enumerate(rows, start=offset):
        caption = row.get("caption", "")[:80]
        url = row.get("url", "n/a")
        sim = row.get("similarity")
        dims = (row.get("width"), row.get("height"))
        print(f"[{idx}] sim={sim:.3f} | {dims} | {caption} | {url}")


def save_image_bytes(image_bytes: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)
    if Image is not None:
        try:
            Image.open(path).verify()
        except Exception:
            pass


def export_images(ds, ids, output_dir="./laion_images"):
    rows = ds.take(ids, columns=["image", "caption", "url"]).to_pylist()
    print(f"\nExporting {len(rows)} images to {output_dir}/ ...")
    for idx, row in zip(ids, rows):
        image_bytes = row.get("image")
        if image_bytes is None:
            continue
        filename = Path(output_dir) / f"{idx:07d}.jpg"
        save_image_bytes(image_bytes, filename)


def find_similar_images(ds, image_index=0, k=5, nprobes=1, refine_factor=1):
    emb_field = ds.schema.field("img_emb")
    ref = ds.take([image_index], columns=["img_emb", "caption", "url"]).to_pylist()[0]
    query = pa.array([ref["img_emb"]], type=emb_field.type)

    results = ds.scanner(
        nearest={
            "column": emb_field.name,
            "q": query[0],
            "k": k + 1,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
        },
        columns=["caption", "url", "similarity"],
    ).to_table().to_pylist()

    print(f"\nQuery image {image_index}: {ref.get('caption', '')[:80]}")
    print(f"Top {k} similar images:")
    for rank, row in enumerate(results[1:k + 1], start=1):
        print(f"  {rank}. sim={row.get('similarity'):.3f} | {row.get('caption', '')[:80]}")
        print(f"     URL: {row.get('url')}")


def filter_quality(ds, min_similarity=0.35, limit=5):
    rows = ds.scanner(
        columns=["caption", "url", "similarity", "NSFW"],
        filter=f"similarity >= {float(min_similarity)} AND \"NSFW\" = 0",
        limit=limit,
    ).to_table().to_pylist()

    print(f"\nHigh-similarity safe rows (>= {min_similarity}):")
    for row in rows:
        print(f"  {row['similarity']:.3f} | {row['caption'][:80]} | {row['url']}")


if __name__ == "__main__":
    print("\nStreaming sample rows via datasets.load_dataset...")
    hf_ds = load_using_hf()
    get_hf_stream_batch(hf_ds)

    print("\nLoading Lance dataset (IVF_PQ index bundled; run queries locally for best perf)...")
    ds = load_dataset()

    show_metadata(ds, limit=3, offset=0)
    filter_quality(ds, min_similarity=0.4, limit=3)
    export_images(ds, ids=[0, 1, 2])
    find_similar_images(ds, image_index=42, k=5)
