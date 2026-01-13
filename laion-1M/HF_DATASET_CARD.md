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
pretty_name: laion-subset-lance
size_categories:
- 1M<n<10M
---
# LAION-Subset (Lance Format)

A lance dataset of LAION image-text corpus (~1M rows) with inline JPEG bytes, CLIP embeddings (`img_emb`), and full metadata available directly from the Hub: `hf://datasets/lance-format/laion-subset`.


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
    "lance-format/laion-subset",
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

ds = lance.dataset("hf://datasets/lance-format/laion-subset")
print(ds.count_rows())
```

> **⚠️ HuggingFace Streaming Note**
> - Streaming is great for sampling metadata but not for ANN queries or bulk image export.
> - Download the dataset locally (`huggingface-cli download lance-format/laion-subset --repo-type dataset --local-dir ./laion`) for heavy usage, then point Lance at `./laion` to use the IVF_PQ index without Hub rate limits.


## Why Lance?

- Optimized for AI workloads: Lance keeps multimodal data and vector search-ready storage in the same columnar format designed for accelerator-era retrieval (see [lance.org](https://lance.org)).
- Images + embeddings + metadata travel as one tabular dataset.
- On-disk, scalable ANN index
- Schema evolution lets you add new features/columns (moderation tags, embeddings, etc.) without rewriting the raw data.

## Quick Start (Lance)

```python
import lance
import pyarrow as pa

lance_ds = lance.dataset("hf://datasets/lance-format/laion-subset")

# Vector search via img_emb IVF_PQ index
emb_field = lance_ds.schema.field("img_emb")
ref = lance_ds.take([0], columns=["img_emb"]).to_pylist()[0]
query = pa.array([ref["img_emb"]], type=emb_field.type)

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

## Dataset Evolution

Want to add new annotations (COCO tags, moderation flags, new embeddings) without rewriting everything? Lance supports transactional schema evolution ([docs](https://lance.org/guide/data_evolution/)).

```python
import lance
import pyarrow as pa

ds = lance.dataset("./laion_subset_local")
ds.add_columns(pa.field("moderation_label", pa.string()))
ds.add_columns({"moderation_label": "case WHEN \"NSFW\" > 0.5 THEN 'review' ELSE 'ok' END"})
```

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
