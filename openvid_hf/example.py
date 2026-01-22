#!/usr/bin/env python3
"""
OpenVid Lance Dataset - Simple Examples

Load from HuggingFace and demonstrate:
1. Blob API - recovering videos from blobs
2. Vector search with nprobes and refine_factor
3. Full-text search on captions
"""

from pathlib import Path

import av
import lance
import lancedb
import datasets
import pyarrow as pa


# ============================================================================
# 1. Load Lance dataset directly from hf in stream model
# ============================================================================

def load_using_hf():
    """Load dataset from HF Hub in streaming mode."""
    download_config = datasets.DownloadConfig(storage_options={"hf": {}})
    info = datasets.get_dataset_config_info(
        "lance-format/openvid-lance",
        download_config=download_config,
    )
    features = info.features.copy()
    features["video_blob"] = datasets.Features({
        "position": datasets.Value("uint64"),
        "size": datasets.Value("uint64"),
    })
    ds = datasets.load_dataset(
        "lance-format/openvid-lance",
        split="train",
        streaming=True,
        features=features,
    )
    print(ds)
    return ds


def get_hf_stream_batch(ds, batch_size=5):
    """Consume a batch from the Hugging Face IterableDataset"""
    rows = list(ds.take(batch_size))
    return pa.Table.from_pylist(rows)


def load_dataset():
    ds = lance.dataset("hf://datasets/lance-format/openvid-lance/data/train.lance")
    print(f"✓ Loaded {ds.count_rows()} videos")
    return ds

def load_lancedb_table():
    db = lancedb.connect("hf://datasets/lance-format/openvid-lance/data")
    tbl = db.open_table("train")
    print(f"✓ Loaded LanceDB table with {len(tbl)} videos")
    return tbl


# ============================================================================
# 2. BLOB API - Recovering Videos
# ============================================================================

def save_video_blob(blob_bytes: bytes, output_path: str):
    """Save video blob to disk"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(blob_bytes)
    print(f"✓ Saved: {output_path} ({len(blob_bytes) / 1024 / 1024:.2f} MB)")


def get_videos_from_batch(ds, limit=10, offset=0):
    """Get metadata and video blobs for a batch of videos."""
    metadata = ds.scanner(
        columns=["caption", "aesthetic_score", "video_path"],
        limit=limit,
        offset=offset
    ).to_table().to_pylist()
    
    print(f"\nBatch metadata (rows {offset}-{offset+limit}):")
    for i, meta in enumerate(metadata):
        print(f"  [{i}] {meta['caption'][:50]}... (score: {meta['aesthetic_score']:.2f})")
    
    print(f"\nLoading video blobs...")
    indices = list(range(offset, offset + len(metadata)))
    blob_files = ds.take_blobs("video_blob", ids=indices)
    blobs = [blob.read() for blob in blob_files]
    
    return blobs, metadata


def export_batch_videos(ds, output_dir="./videos", limit=5, offset=0):
    """Export a batch of videos to disk"""
    blobs, metadata = get_videos_from_batch(ds, limit=limit, offset=offset)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting videos to {output_dir}/...")
    for i, (blob_bytes, meta) in enumerate(zip(blobs, metadata)):
        filename = f"{offset + i:06d}.mp4"
        output_path = Path(output_dir) / filename
        save_video_blob(blob_bytes, str(output_path))


def inspect_video_with_pyav(ds, video_index=0):
    """Seek within a blob and print the first frame past each timestamp."""
    print(f"\nInspecting video index {video_index} with PyAV")
    blob_file = ds.take_blobs("video_blob", ids=[video_index])[0]

    with av.open(blob_file) as container:
        stream = container.streams.video[0]
        for seconds in (0.0, 1.0, 2.5):
            target_pts = int(seconds / stream.time_base)
            container.seek(target_pts, stream=stream)
            frame = next((f for f in container.decode(stream) if f.time is not None and f.time >= seconds), None)
            if frame:
                print(f"  Seek {seconds:.1f}s -> {frame.width}x{frame.height} (pts={frame.pts}, time={frame.time:.2f}s)")
            else:
                print(f"  Seek {seconds:.1f}s -> no frame decoded")


if __name__ == "__main__":
    # 1. Load dataset from HuggingFace Hub
    print("\nLoading dataset using hf and getting a batch...")
    hf_ds = load_using_hf()
    batch = get_hf_stream_batch(hf_ds)
    print(batch)

    print("\nLoading full dataset using lance")
    ds = load_dataset()
    
    # ============================================================================
    # LANCE EXAMPLES
    # ============================================================================
    print("\n" + "="*30 + " LANCE EXAMPLES " + "="*30)

    print("\n" + "="*70)
    print("EXAMPLE 1: Blob API - Export Batch of Videos")
    export_batch_videos(ds, output_dir="./example_videos", limit=3, offset=0)
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Inspecting Indices")
    print(ds.list_indices())

    print("\n" + "="*70)
    print("EXAMPLE 3: Vector Search with nprobes=1")
    ref_video = ds.take([100], columns=["embedding", "caption"]).to_pylist()[0]
    print(f"\nQuery video: {ref_video['caption']}")
    query_vector = pa.array([ref_video['embedding']], type=pa.list_(pa.float32(), 1024))
    results = ds.scanner(
        nearest={"column": "embedding", "q": query_vector[0], "k": 6, "nprobes": 1}
    ).to_table().to_pylist()
    print(f"Top 5 similar videos:")
    for i, video in enumerate(results[1:], 1):
        print(f"  {i}. {video['caption'][:60]}... (Aesthetic: {video['aesthetic_score']:.2f})")

    print("\n" + "="*70)
    print("EXAMPLE 4: Full-Text Search (Lance Native FTS)")
    query = "sunset"
    print(f"\nSearching for: '{query}'")
    results = ds.scanner(
        full_text_query=query,
        columns=["caption", "aesthetic_score"],
        limit=2,
        fast_search=True
    ).to_table().to_pylist()
    print(f"Found {len(results)} results:")
    for i, video in enumerate(results, 1):
        print(f"  {i}. {video['caption'][:60]}... (Quality: aesthetic={video['aesthetic_score']:.2f})")

    print("\n" + "="*70)
    print("EXAMPLE 5: PyAV Decode & Seeks")
    inspect_video_with_pyav(ds, video_index=3500)

    # ============================================================================
    # LANCE DB EXAMPLES
    # ============================================================================
    print("\n" + "="*30 + " LANCEDB EXAMPLES " + "="*30)

    print("\nLoading dataset using lancedb")
    tbl = load_lancedb_table()

    print("\n" + "="*70)
    print("LANCEDB EXAMPLE 1: Vector Search with nprobes=1")
    ref_video_db = tbl.search().limit(1).select(["embedding", "caption"]).to_pandas().to_dict('records')[0]
    print(f"\nQuery video: {ref_video_db['caption']}")
    results_db = tbl.search(ref_video_db['embedding']).limit(5).nprobes(1).to_list()
    print(f"Top 5 similar videos:")
    for i, video in enumerate(results_db[1:], 1):
        print(f"  {i}. {video['caption'][:60]}... (Aesthetic: {video['aesthetic_score']:.2f})")

    print("\n" + "="*70)
    print("LANCEDB EXAMPLE 2: Full-Text Search (LanceDB FTS)")
    query_db = "sunset"
    print(f"\nSearching for: '{query_db}'")
    results_db_fts = tbl.search(query_db).select(["caption", "aesthetic_score"]).limit(2).to_list()
    print(f"Found {len(results_db_fts)} results:")
    for i, video in enumerate(results_db_fts, 1):
        print(f"  {i}. {video['caption'][:60]}... (Quality: aesthetic={video['aesthetic_score']:.2f})")
