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
import pyarrow as pa


# ============================================================================
# 1. Load Lance dataset directly from hf in stream model
# ============================================================================

def load_dataset():
    ds = lance.dataset("hf://datasets/lance-format/openvid-lance")
    print(f"✓ Loaded {ds.count_rows()} videos")
    return ds


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
    scanner = ds.scanner(
        columns=["caption", "aesthetic_score", "video_path"],
        limit=limit,
        offset=offset
    )
    metadata = scanner.to_table().to_pylist()
    
    print(f"\nBatch metadata (rows {offset}-{offset+limit}):")
    for i, meta in enumerate(metadata):
        print(f"  [{i}] {meta['caption'][:50]}... (score: {meta['aesthetic_score']:.2f})")
    
    # Now load blobs for selected videos using take_blobs()
    print(f"\nLoading video blobs...")
    indices = list(range(offset, offset + len(metadata)))
    
    # Use take_blobs() to get BlobFile objects
    blob_files = ds.take_blobs("video_blob", ids=indices)
    
    # Read bytes from BlobFile objects
    blobs = []
    for blob_file in blob_files:
        blob_bytes = blob_file.read()
        blobs.append(blob_bytes)
    
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

            frame = None
            for candidate in container.decode(stream):
                if candidate.time is None:
                    continue
                frame = candidate
                if frame.time >= seconds:
                    break

            if frame is None:
                print(f"  Seek {seconds:.1f}s -> no frame decoded")
                continue

            print(
                f"  Seek {seconds:.1f}s -> {frame.width}x{frame.height} "
                f"(pts={frame.pts}, time={frame.time:.2f}s)"
            )


# ============================================================================
# 3. VECTOR SEARCH
# ============================================================================

def vector_search(ds, query_embedding, k=10, nprobes=1, refine_factor=1):
    query_vector = pa.array([query_embedding], type=pa.list_(pa.float32(), 1024))
    
    results = ds.scanner(
        nearest={
            "column": "embedding",
            "q": query_vector[0],
            "k": k,
            "nprobes": nprobes,
            "refine_factor": refine_factor
        }
    ).to_table().to_pylist()
    
    return results


def find_similar_videos(ds, video_index, k=5, nprobes=1):
    ref_table = ds.take([video_index], columns=["embedding", "caption"])
    ref = ref_table.to_pylist()[0]
    
    print(f"\nQuery video: {ref['caption']}")
    print(f"Searching with nprobes={nprobes}...\n")
    
    results = vector_search(ds, ref['embedding'], k=k+1, nprobes=nprobes)
    
    similar = results[1:k+1]
    
    print(f"Top {k} similar videos:")
    for i, video in enumerate(similar, 1):
        print(f"  {i}. {video['caption'][:60]}...")
        print(f"     Aesthetic: {video['aesthetic_score']:.2f}")
    
    return similar


# ============================================================================
# 4. FULL-TEXT SEARCH (Lance Native FTS)
# ============================================================================

def fts_search(ds, query_text, limit=10):
    #Full-text search on captions using Lance's native FTS
    results = ds.scanner(
        full_text_query=query_text,
        columns=["caption", "aesthetic_score", "motion_score"],
        limit=limit,
        fast_search=True 
    ).to_table().to_pylist()
    
    return results


def search_captions(ds, query, limit=5):
    print(f"\nSearching for: '{query}'")
    
    results = fts_search(ds, query, limit=limit)
    
    print(f"Found {len(results)} results:")
    for i, video in enumerate(results, 1):
        print(f"  {i}. {video['caption'][:60]}...")
        print(f"     Quality: aesthetic={video['aesthetic_score']:.2f}")
    
    return results


if __name__ == "__main__":
    ds = load_dataset()
    
    print("="*70)
    print("EXAMPLE 1: Blob API - Export Batch of Videos")
    export_batch_videos(ds, output_dir="./example_videos", limit=3, offset=0)
    
    print("="*70)
    print("EXAMPLE 2: Vector Search with nprobes=1")
    similar = find_similar_videos(ds, video_index=100, k=5, nprobes=1)
    
    print("="*70)
    print("EXAMPLE 3: Full-Text Search (Lance Native FTS)")
    results = search_captions(ds, "sunset", limit=2)

    print("="*70)
    print("EXAMPLE 4: PyAV Decode & Seeks")
    inspect_video_with_pyav(ds, video_index=3500)
    
