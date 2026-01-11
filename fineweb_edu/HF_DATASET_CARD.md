---
license: cc-by-4.0
task_categories:
- text-retrieval
- question-answering
- lance
language:
- en
tags:
- retrieval
- text
- lance
pretty_name: fineweb-edu-lance
size_categories:
- 1M<n<10M
---
# FineWeb-Edu (Lance Format)

FineWeb-edu dataset with over 1.5 billion rows. Each passage ships with cleaned text, metadata, and 384-dim text embeddings for retrieval-heavy workloads.


## Load via `datasets.load_dataset`

```python
import datasets
import pyarrow as pa

hf_ds = datasets.load_dataset(
    "lance-format/fineweb-edu",
    split="train",
    streaming=True,
)

batch_rows = list(hf_ds.take(5))
batch_table = pa.Table.from_pylist(batch_rows)
batch_df = batch_table.to_pandas()

print(batch_table)
print(batch_df.head())
```

Use Lance's native connector when you need ANN search, FTS, or direct access to embeddings while still pointing to the copy hosted on Hugging Face:

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/fineweb-edu")
print(f"Total passages: {ds.count_rows():,}")
```


> **Index Status & Streaming Guidance**
> - Pre-built ANN/FTS indexes aren't uploaded yet— It is recommended todownload the dataset locally and build indexes yourself before running similarity/search demos. 
> - The corpus is large (~1.5B passages). Heavy retrieval workloads should point Lance at a local copy.

## Why Lance?

- Vector + FTS indices live with the dataset, so similarity search is a single API call.
- Columnar format keeps metadata scans fast even when the corpus spans billions of tokens.
- Snapshotting & schema evolution make it easy to add new annotations (`text_embedding`, moderation tags, etc.) without rewriting raw text.
- The `hf://` URI surfaces Hub-hosted data in any Lance runtime (Python, Rust, node, DuckDB).

## Quick Start (Lance Python)

```python
import lance
import pyarrow as pa

lance_ds = lance.dataset("hf://datasets/lance-format/fineweb-edu")

# Browse titles & language without touching embeddings
rows = lance_ds.scanner(
    columns=["title", "language"],
    limit=5
).to_table().to_pylist()

# Vector similarity from the on-dataset ANN index
ref = lance_ds.take([0], columns=["text_embedding", "title"])
query_vec = pa.array([ref.to_pylist()[0]["text_embedding"]],
                     type=ref.schema.field("text_embedding").type)

results = lance_ds.scanner(
    nearest={
        "column": "text_embedding",
        "q": query_vec[0],
        "k": 5,
        "nprobes": 8,
        "refine_factor": 20,
    },
    columns=["title", "language", "text"],
).to_table().to_pylist()
```

> **Hugging Face Streaming Note**
> - Streaming uses conservative ANN parameters (`nprobes`, `refine_factor`) to stay within HF rate limits.
> - Prefer local copies (`huggingface-cli download lance-format/fineweb-edu --local-dir ./fineweb`) for heavy workloads, then point Lance at `./fineweb`.

## Dataset Schema

Common columns you'll find in this Lance export:
- `text` – cleaned passage content.
- `title` – page/article title when available.
- `url` – canonical source URL.
- `language` + `language_probability` – detector outputs for filtering.
- Quality metadata from FineWeb-Edu (e.g., heuristic scores or length stats).
- `text_embedding` – 384-dimension float32 vector for retrieval.

## Usage Examples

> **Reference-only search snippets**
> The vector/FTS examples below show the Lance APIs you’ll use once indexes are available. The hosted dataset doesn’t yet ship ANN/FTS indexes—download locally (or build indexes yourself) before running them. Pre-built indexes are coming soon.

### 1. Sample documents without embeddings

```python
scanner = ds.scanner(
    columns=["title", "language", "text"],
    filter="language = 'en'",
    limit=5,
)
for doc in scanner.to_table().to_pylist():
    print(doc["title"], doc["language"])
    print(doc["text"][:200], "...\n")
```

### 2. Vector search for semantically similar passages

```python
ref_doc = ds.take([123], columns=["text_embedding", "title", "text"]).to_pylist()[0]
emb_type = ds.to_table(columns=["text_embedding"], limit=1).schema.field("text_embedding").type
query = pa.array([ref_doc["text_embedding"]], type=emb_type)

neighbors = ds.scanner(
    nearest={
        "column": "text_embedding",
        "q": query[0],
        "k": 6,
        "nprobes": 8,
        "refine_factor": 20,
    },
    columns=["title", "language", "text"],
).to_table().to_pylist()[1:]
```

### 3. Full-text search with Lance FTS

```python
hits = ds.scanner(
    full_text_query="quantum computing",
    columns=["title", "language", "text"],
    limit=10,
    fast_search=True,
).to_table().to_pylist()
```


See `fineweb_edu/example.py` on lance-huggingface repo for a complete walkthrough that combines HF streaming batches with Lance-powered retrieval.

## Dataset Evolution

Lance datasets can evolve without full rewrites—perfect for FineWeb-Edu as new signals arrive. Add schema-only columns, backfill with SQL or Python, merge offline annotations, and rename/cast fields while keeping previous snapshots queryable ([docs](https://lance.org/guide/data_evolution)). The snippet below mirrors the official Lance docs so you can adapt it to your own metadata or embedding workflows.

```python
import lance
import pyarrow as pa
import numpy as np

base = pa.table({"id": pa.array([1, 2, 3]), "text": pa.array(["A", "B", "C"])})
dataset = lance.write_dataset(base, "fineweb_evolution", mode="overwrite")

# 1. Add placeholder metadata
dataset.add_columns(pa.field("subject", pa.string()))

# 2. Fill via SQL expressions
dataset.add_columns({"quality_bucket": "'unknown'"})

# 3. Generate embeddings with a batch UDF
@lance.batch_udf()
def random_embedding(batch):
    vecs = np.random.rand(batch.num_rows, 384).astype("float32")
    return pa.RecordBatch.from_arrays(
        [pa.FixedSizeListArray.from_arrays(vecs.ravel(), 384)],
        names=["text_embedding"],
    )

dataset.add_columns(random_embedding)

# 4. Merge offline annotations
labels = pa.table({"id": pa.array([1, 2, 3]), "label": pa.array(["math", "history", "science"])})
dataset.merge(labels, "id")

# 5. Rename / cast columns without touching others
dataset.alter_columns({"path": "subject", "name": "topic"})
dataset.alter_columns({"path": "text_embedding", "data_type": pa.list_(pa.float16(), 384)})
```

These operations are metadata-aware and snapshot-safe, so you can iterate on embeddings, quality tags, or moderation fields while keeping earlier dataset versions available for reproducible experiments.
