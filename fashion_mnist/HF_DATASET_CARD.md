---
license: mit
task_categories:
- image-classification
- image-feature-extraction
language:
- en
tags:
- fashion-mnist
- lance
- clip-embeddings
pretty_name: fashion-mnist-lance
size_categories:
- 10K<n<100K
---
# Fashion-MNIST (Lance Format)

A Lance-formatted version of [Fashion-MNIST](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) with **70,000 28×28 grayscale clothing images** stored inline alongside CLIP embeddings and a pre-built `IVF_PQ` ANN index.

## Key features

- All multimodal data (image bytes + embeddings) stored **inline** in the same Lance dataset.
- **Pre-computed CLIP embeddings** (OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k`, 512-dim, L2-normalized) with an `IVF_PQ` index.
- **BTREE on `label`** and **BITMAP on `label_name`** for fast filtered scans.

## Splits

| Split | Rows |
|-------|------|
| `train` | 60,000 |
| `test`  | 10,000 |

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | `int64` | Row index within the split |
| `image` | `large_binary` | Inline PNG bytes (28×28 grayscale) |
| `label` | `int32` | Class id (0-9) |
| `label_name` | `string` | One of `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle_boot` |
| `image_emb` | `fixed_size_list<float32, 512>` | CLIP image embedding (cosine-normalized) |

## Pre-built indices

- `IVF_PQ` on `image_emb` — `metric=cosine`
- `BTREE` on `label`
- `BITMAP` on `label_name`

## Load with Lance

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/fashion-mnist-lance/data/train.lance")
print(ds.count_rows(), ds.schema.names, ds.list_indices())
```

## Load with `datasets.load_dataset`

```python
import datasets

hf_ds = datasets.load_dataset("lance-format/fashion-mnist-lance", split="train", streaming=True)
for row in hf_ds.take(3):
    print(row["label_name"])
```

> **Tip — for production use, download locally first** to avoid Hub rate limits:
> ```bash
> hf download lance-format/fashion-mnist-lance --repo-type dataset --local-dir ./fashion-mnist-lance
> ```

## Vector search example

```python
import lance
import pyarrow as pa

ds = lance.dataset("hf://datasets/lance-format/fashion-mnist-lance/data/train.lance")
emb_field = ds.schema.field("image_emb")
ref = ds.take([0], columns=["image_emb"]).to_pylist()[0]["image_emb"]
query = pa.array([ref], type=emb_field.type)

neighbors = ds.scanner(
    nearest={"column": "image_emb", "q": query[0], "k": 5, "nprobes": 16, "refine_factor": 30},
    columns=["id", "label_name"],
).to_table().to_pylist()
```

## Filter by class

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/fashion-mnist-lance/data/train.lance")
sneakers = ds.scanner(filter="label_name = 'Sneaker'", columns=["id"], limit=5).to_table()
```

## Why Lance?

- One dataset for images + embeddings + indices + metadata — no sidecar files.
- On-disk vector and FTS indices live next to the data, so search works on local copies and the Hub.
- Schema evolution: add new columns (model predictions, fresh embeddings, augmentations) without rewriting the data.

## Source & license

Converted from [`zalando-datasets/fashion_mnist`](https://huggingface.co/datasets/zalando-datasets/fashion_mnist). Released under the MIT license.

## Citation

```
@online{xiao2017fashionmnist,
  title={Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  year={2017},
  eprint={1708.07747},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
