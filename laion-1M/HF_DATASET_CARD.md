---
license: cc-by-4.0
task_categories:
- image-retrieval
- multi-modal-retrieval
- lance
language:
- en
tags:
- laion
- clip
- vision-language
- lance
pretty_name: laion-1m-lance
size_categories:
- 1M<n<10M
---
# LAION-Subset (Lance Format)

A lance dataset of LAION image-text corpus (~1M rows) with inline JPEG bytes, CLIP embeddings (`img_emb`), and full metadata available directly from the Hub: `hf://datasets/lance-format/laion-1m/data/train.lance`.


## Key Features

- **Images stored inline** – the `image` column is binary data, so sampling/exporting images never leaves Lance.
- **Prebuilt ANN index** – `img_emb` ships with IVF_PQ for instant similarity search.
- **Metadata rich** – captions, URLs, NSFW flags, EXIF, dimensions, similarity scores, etc.
- **Lance<>HF integration** – access via `datasets` or connect with Lance for ANN search, image export, and any operation that needs the vector index or binary blobs.

## Load with `datasets.load_dataset`

```python
import datasets
import pandas as pd

hf_ds = datasets.load_dataset(
    "lance-format/laion-1m",
    split="train",
    streaming=True,
)
batch = list(hf_ds.take(5))
print(pd.DataFrame.from_records(batch).head())
```

## Load with Lance

Use Lance for ANN search, image export, and any operation that needs the vector index or binary blobs:

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")
print(ds.count_rows())
```

These tables can also be consumed by [LanceDB](https://lancedb.github.io/lancedb/), the serverless vector database built on Lance, for simplified vector search and other queries.

```python
import lancedb

db = lancedb.connect("hf://datasets/lance-format/laion-subset/data")
tbl = db.open_table("train")
print(f"LanceDB table opened with {len(tbl)} image-text pairs")
```

> **⚠️ HuggingFace Streaming Note**
> - Download the dataset locally (`huggingface-cli download lance-format/laion-1m --repo-type dataset --local-dir ./laion`) for heavy usage, then point Lance at `./laion` to use the IVF_PQ index.


## Why Lance?

- Optimized for AI workloads: Lance keeps multimodal data and vector search-ready storage in the same columnar format designed for accelerator-era retrieval (see [lance.org](https://lance.org)).
- Images + embeddings + metadata travel as one tabular dataset.
- On-disk, scalable ANN index
- Schema evolution lets you add new features/columns (moderation tags, embeddings, etc.) without rewriting the raw data.

## Quick Start (Lance)

### Inspecting Existing Indices

This dataset comes with a built in vector (IVF) index for image embeddings. You can inspect the prebuilt indices on the dataset:

```python
import lance

dataset = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

# List all indices
indices = dataset.list_indices()
print(indices)
```

While this dataset comes with pre-built indices, you can also create your own custom indices if needed. For example:

```python
# ds is a local Lance dataset
ds.create_index(
    "img_emb",
    index_type="IVF_PQ",
    num_partitions=256,
    num_sub_vectors=96,
    replace=True,
)
```

```python
# ds is a local Lance dataset
ds.create_fts_index("caption")
```

## Quick Start (Lance)

```python
import lance
import pyarrow as pa

lance_ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

# Vector search via img_emb IVF_PQ index
emb_field = lance_ds.schema.field("img_emb")
query = pa.array(list(range(768)), type=emb_field.type)

neighbors = lance_ds.scanner(
    nearest={
        "column": emb_field.name,
        "q": query[0],
        "k": 6,
        "nprobes": 16,
        "refine_factor": 30,
    },
    columns=["caption", "url", "similarity"],
).to_table().to_pylist()
```

## Storing & Retrieving Multimodal Data

```python
from pathlib import Path

rows = lance_ds.take([0, 1], columns=["image", "caption"]).to_pylist()
for idx, row in enumerate(rows):
    Path("samples").mkdir(exist_ok=True)
    with open(f"samples/{idx}.jpg", "wb") as f:
        f.write(row["image"])
```

Images are stored inline as binary columns (regular Lance binary, not the special blob handle used in OpenVid). They behave like any other column—scan captions without touching `image`, then `take()` when you want the bytes.
```

## Dataset Schema

Core fields:
- `image_path`, `image`
- `caption`, `url`
- `NSFW` (uppercase), `similarity`, `LICENSE`, `key`, `status`, `error_message`
- `width`, `height`, `original_width`, `original_height`
- `exif`, `md5`
- `img_emb`


## Usage Examples

### 1. Browse metadata

```python
scanner = ds.scanner(columns=["caption", "url", "similarity"], limit=5)
for row in scanner.to_table().to_pylist():
    print(row)
```

### 2. Export images

```python
rows = ds.take(range(3), columns=["image", "caption"]).to_pylist()
for i, row in enumerate(rows):
    with open(f"sample_{i}.jpg", "wb") as f:
        f.write(row["image"])
```

### 3. Vector similarity search

```python
emb_field = ds.schema.field("img_emb")
ref = ds.take([123], columns=["img_emb"]).to_pylist()[0]
query = pa.array([ref["img_emb"]], type=emb_field.type)

neighbors = ds.scanner(
    nearest={
        "column": emb_field.name,
        "q": query[0],
        "k": 6,
        "nprobes": 16,
        "refine_factor": 30,
    },
    columns=["caption", "url", "similarity"],
).to_table().to_pylist()
```

### LanceDB Vector Similarity Search

```python
import lancedb

db = lancedb.connect("hf://datasets/lance-format/laion-1m/data")
query_embedding = list(range(768))

results = tbl.search(query_embedding) \
    .limit(5) \
    .to_list()

```

### LanceDB Full-Text Search

```python
import lancedb

db = lancedb.connect("hf://datasets/lance-format/laion-1m/data")
tbl = db.open_table("train")

results = tbl.search("dog running") \
    .select(["caption", "url", "similarity"]) \
    .limit(10) \
    .to_list()
```

## Dataset Evolution

Lance supports flexible schema and data evolution ([docs](https://lance.org/guide/data_evolution/)). You can add/drop columns, backfill with SQL or Python, rename fields, or change data types without rewriting the whole dataset. In practice this lets you:
- Introduce fresh metadata (moderation labels, embeddings, quality scores) as new signals become available.
- Add new columns to existing datasets without re-exporting terabytes of video.
- Adjust column names or shrink storage (e.g., cast embeddings to float16) while keeping previous snapshots queryable for reproducibility.

```python
import lance
import pyarrow as pa
import numpy as np

# Assume ds is a local Lance dataset
# ds = lance.dataset("./laion_1m_local")

base = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(base, "laion_evolution", mode="overwrite")

# 1. Grow the schema instantly (metadata-only)
dataset.add_columns(pa.field("quality_bucket", pa.string()))

# 2. Backfill with SQL expressions or constants
dataset.add_columns({"status": "'active'"})

# 3. Generate rich columns via Python batch UDFs
@lance.batch_udf()
def random_embedding(batch):
    arr = np.random.rand(batch.num_rows, 128).astype("float32")
    return pa.RecordBatch.from_arrays(
        [pa.FixedSizeListArray.from_arrays(arr.ravel(), 128)],
        names=["embedding"],
    )

dataset.add_columns(random_embedding)

# 4. Bring in offline annotations with merge
labels = pa.table({
    "id": pa.array([1, 2, 3]),
    "label": pa.array(["horse", "rabbit", "cat"]),
})
dataset.merge(labels, "id")

# 5. Rename or cast columns as needs change
dataset.alter_columns({"path": "quality_bucket", "name": "quality_tier"})
dataset.alter_columns({"path": "embedding", "data_type": pa.list_(pa.float16(), 128)})
```

These operations are automatically versioned, so prior experiments can still point to earlier versions while the dataset keeps evolving.

## Citation

```
@article{schuhmann2022laion5b,
  title={LAION-5B: An open large-scale dataset for training next generation image-text models},
  author={Schuhmann, Christoph and others},
  journal={NeurIPS Datasets and Benchmarks Track},
  year={2022}
}
```

## License

Content inherits LAION’s original licensing and safety guidelines. Review [LAION policy](https://laion.ai/blog/laion-5b/) before downstream use.
