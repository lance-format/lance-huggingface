---
license: cc-by-4.0
task_categories:
- text-to-video
- video-classification
- lance
language:
- en
tags:
- text-to-video
- video-search
pretty_name: openvid-lance
size_categories:
- 100K<n<1M
---
# OpenVid Dataset (Lance Format)

Lance format version of the [OpenVid dataset](https://huggingface.co/datasets/nkp37/OpenVid-1M) with **937,957 high-quality videos** stored with inline video blobs, embeddings, and rich metadata.

![](https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid-1M.png)

**Key Features:**
The dataset is stored in lance format with inline video blobs, video embeddings, and rich metadata.

- **Videos stored inline as blobs** - No external files to manage
- **Efficient column access** - Load metadata without touching video data
- **Prebuilt indices available** - IVF_PQ index for similarity search, FTS index on captions
- **Fast random access** - Read any video instantly by index
- **HuggingFace integration** - Load directly from the Hub in streaming mode

## Lance Blob API

Lance stores videos as **inline blobs** - binary data embedded directly in the dataset. This provides:

- **Single source of truth** - Videos and metadata together in one dataset
- **Lazy loading** - Videos only loaded when you explicitly request them
- **Efficient storage** - Optimized encoding for large binary data
- **Transactional consistency** - Query and retrieve in one atomic operation


```python
import lance

ds = lance.dataset("hf://datasets/lance-format/openvid-lance")

# 1. Browse metadata without loading video data
metadata = ds.scanner(
    columns=["caption", "aesthetic_score"],  # No video_blob column!
    filter="aesthetic_score >= 4.5",
    limit=10
).to_table().to_pylist()

# 2. User selects video to watch
selected_index = 3

# 3. Load only that video blob
blob_file = ds.take_blobs("video_blob", ids=[selected_index])[0]
video_bytes = blob_file.read()

# 4. Save to disk
with open("video.mp4", "wb") as f:
    f.write(video_bytes)
```

## Quick Start

```python
import lance

# Load dataset from HuggingFace
ds = lance.dataset("hf://datasets/lance-format/openvid-lance")
print(f"Total videos: {ds.count_rows():,}")
```

> **⚠️ HuggingFace Streaming Note**
> 
> When streaming from HuggingFace (as shown above), some operations use minimal parameters to avoid rate limits:
> - `nprobes=1` for vector search (lowest value)
> - Column selection to reduce I/O
> 
> **You may still hit rate limits on HuggingFace's free tier.** For best performance and to avoid rate limits, **download the dataset locally**:
> 
> ```bash
> # Download once
> huggingface-cli download lance-format/openvid-lance --repo-type dataset --local-dir ./openvid
> 
> # Then load locally
> ds = lance.dataset("./openvid")
> ```
> 
> Streaming is recommended only for quick exploration and testing.

## Dataset Schema

Each row contains:
- `video_blob` - Video file as binary blob (inline storage)
- `caption` - Text description of the video
- `embedding` - 1024-dim vector embedding
- `aesthetic_score` - Visual quality score (0-5+)
- `motion_score` - Amount of motion (0-1)
- `temporal_consistency_score` - Frame consistency (0-1)
- `camera_motion` - Camera movement type (pan, zoom, static, etc.)
- `fps`, `seconds`, `frame` - Video properties

## Usage Examples

### 1. Browse Metadata quickly (Fast - No Video Loading)

```python
# Load only metadata without heavy video blobs
scanner = ds.scanner(
    columns=["caption", "aesthetic_score", "motion_score"],
    limit=10
)
videos = scanner.to_table().to_pylist()

for video in videos:
    print(f"{video['caption']} - Quality: {video['aesthetic_score']:.2f}")
```

### 2. Export Videos from Blobs

```python
# Load specific videos by index
indices = [0, 100, 500]
blob_files = ds.take_blobs("video_blob", ids=indices)

# Save to disk
for i, blob_file in enumerate(blob_files):
    with open(f"video_{i}.mp4", "wb") as f:
        f.write(blob_file.read())
```

### 3. Open inline videos with PyAV and run seeks directly on the blob file

```python
import av

selected_index = 123
blob_file = ds.take_blobs("video_blob", ids=[selected_index])[0]

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

        print(
            f"Seek {seconds:.1f}s -> {frame.width}x{frame.height} "
            f"(pts={frame.pts}, time={frame.time:.2f}s)"
        )
```

### 4. Vector Similarity Search

```python
import pyarrow as pa

# Find similar videos
ref_video = ds.take([0], columns=["embedding"]).to_pylist()[0]
query_vector = pa.array([ref_video['embedding']], type=pa.list_(pa.float32(), 1024))

results = ds.scanner(
    nearest={
        "column": "embedding",
        "q": query_vector[0],
        "k": 5,
        "nprobes": 1,
        "refine_factor": 1
    }
).to_table().to_pylist()

for video in results[1:]:  # Skip first (query itself)
    print(video['caption'])
```

### 5. Full-Text Search

```python
# Search captions using FTS index
results = ds.scanner(
    full_text_query="sunset beach",
    columns=["caption", "aesthetic_score"],
    limit=10,
    fast_search=True
).to_table().to_pylist()

for video in results:
    print(f"{video['caption']} - {video['aesthetic_score']:.2f}")
```

### 6. Filter by Quality

```python
# Get high-quality videos
high_quality = ds.scanner(
    filter="aesthetic_score >= 4.5 AND motion_score >= 0.3",
    columns=["caption", "aesthetic_score", "camera_motion"],
    limit=20
).to_table().to_pylist()
```

## Dataset Statistics

- **Total videos**: 937,957
- **Embedding dimension**: 1024
- **Video formats**: MP4 (H.264)
- **Index types**: IVF_PQ (vector),  FTS


## Citation

@article{nan2024openvid,
  title={OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation},
  author={Nan, Kepan and Xie, Rui and Zhou, Penghao and Fan, Tiehan and Yang, Zhenheng and Chen, Zhijie and Li, Xiang and Yang, Jian and Tai, Ying},
  journal={arXiv preprint arXiv:2407.02371},
  year={2024}
}


## License

Please check the original OpenVid dataset license for usage terms.
