---
license: other
task_categories:
- image-classification
- image-feature-extraction
language:
- en
tags:
- stanford-cars
- fine-grained
- cars
- lance
- clip-embeddings
pretty_name: stanford-cars-lance
size_categories:
- 1K<n<10K
---
# Stanford Cars (Lance Format)

Lance-formatted version of the [Stanford Cars dataset](https://web.archive.org/web/20210212183835/http://ai.stanford.edu/~jkrause/cars/car_dataset.html) — 8,144 training images across 196 fine-grained car make/model/year classes — sourced from [`Multimodal-Fatima/StanfordCars_train`](https://huggingface.co/datasets/Multimodal-Fatima/StanfordCars_train).

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | `int64` | Row index |
| `image` | `large_binary` | Inline JPEG bytes |
| `label` | `int32` | Class id (0-195) |
| `blip_caption` | `string?` | BLIP-generated caption (beam=5) carried through from the source mirror |
| `image_emb` | `fixed_size_list<float32, 512>` | OpenCLIP `ViT-B-32` embedding (cosine-normalized) |

## Pre-built indices

- `IVF_PQ` on `image_emb` — `metric=cosine`
- `INVERTED` (FTS) on `blip_caption`
- `BTREE` on `label`

## Quick start

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/stanford-cars-lance/data/train.lance")
print(ds.count_rows(), ds.schema.names, ds.list_indices())
```

## Caption-based filtering

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/stanford-cars-lance/data/train.lance")
hits = ds.scanner(full_text_query="red sports car", columns=["id", "blip_caption"], limit=10).to_table()
```

## Visual similarity search

```python
import lance, pyarrow as pa
ds = lance.dataset("hf://datasets/lance-format/stanford-cars-lance/data/train.lance")
emb_field = ds.schema.field("image_emb")
ref = ds.take([0], columns=["image_emb", "blip_caption"]).to_pylist()[0]
neighbors = ds.scanner(
    nearest={"column": "image_emb", "q": pa.array([ref["image_emb"]], type=emb_field.type)[0], "k": 5},
    columns=["id", "blip_caption"],
).to_table().to_pylist()
```

## Source & license

Converted from [`Multimodal-Fatima/StanfordCars_train`](https://huggingface.co/datasets/Multimodal-Fatima/StanfordCars_train), itself a parquet redistribution of the Stanford Cars test split. The original dataset license is for non-commercial research use; review the [Stanford Cars terms](https://github.com/jhoffman/stanford-cars) before redistribution.

## Citation

```
@inproceedings{krause2013collecting,
  title={Collecting a large-scale dataset of fine-grained cars},
  author={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
  booktitle={Workshop on Fine-Grained Visual Categorization (CVPR)},
  year={2013}
}
```
