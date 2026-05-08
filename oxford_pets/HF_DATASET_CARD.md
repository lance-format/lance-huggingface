---
license: cc-by-sa-4.0
task_categories:
- image-classification
- image-feature-extraction
language:
- en
tags:
- oxford-pets
- fine-grained
- pets
- lance
- clip-embeddings
pretty_name: oxford-pets-lance
size_categories:
- 1K<n<10K
---
# Oxford-IIIT Pet (Lance Format)

Lance-formatted version of the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 7,390 cat & dog photos across 37 breeds — sourced from [`pcuenq/oxford-pets`](https://huggingface.co/datasets/pcuenq/oxford-pets).

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | `int64` | Row index |
| `image` | `large_binary` | Inline JPEG bytes |
| `label_name` | `string` | One of 37 breeds, underscore-spaced (`british_shorthair`, `golden_retriever`, …) |
| `is_dog` | `bool` | `true` for dog breeds, `false` for cat breeds |
| `path` | `string?` | Original filename in the source dataset |
| `image_emb` | `fixed_size_list<float32, 512>` | OpenCLIP `ViT-B-32` embedding (cosine-normalized) |

## Pre-built indices

- `IVF_PQ` on `image_emb` — `metric=cosine`
- `BITMAP` on `label_name` and `is_dog`

## Quick start

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/oxford-pets-lance/data/train.lance")
print(ds.count_rows(), ds.schema.names, ds.list_indices())
```

## Filter — only dogs, only golden retrievers, etc.

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/oxford-pets-lance/data/train.lance")
dogs = ds.scanner(filter="is_dog = true", columns=["label_name"], limit=5).to_table()
goldens = ds.scanner(filter="label_name = 'golden_retriever'", columns=["id"], limit=5).to_table()
```

## Visual similarity search

```python
import lance, pyarrow as pa
ds = lance.dataset("hf://datasets/lance-format/oxford-pets-lance/data/train.lance")
emb_field = ds.schema.field("image_emb")
ref = ds.take([0], columns=["image_emb", "label_name"]).to_pylist()[0]
neighbors = ds.scanner(
    nearest={"column": "image_emb", "q": pa.array([ref["image_emb"]], type=emb_field.type)[0], "k": 5},
    columns=["id", "label_name"],
).to_table().to_pylist()
```

## Source & license

Converted from [`pcuenq/oxford-pets`](https://huggingface.co/datasets/pcuenq/oxford-pets). Released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

```
@inproceedings{parkhi2012cats,
  title={Cats and Dogs},
  author={Parkhi, Omkar M. and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, C. V.},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2012}
}
```
