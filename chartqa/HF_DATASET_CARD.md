---
license: gpl-3.0
task_categories:
- visual-question-answering
- image-text-to-text
language:
- en
tags:
- chartqa
- charts
- vqa
- vision-language
- lance
- clip-embeddings
pretty_name: chartqa-lance
size_categories:
- 1K<n<10K
---
# ChartQA (Lance Format)

Lance-formatted version of [ChartQA](https://github.com/vis-nlp/ChartQA) — VQA over scientific and business charts that combine logical and visual reasoning — sourced from [`lmms-lab/ChartQA`](https://huggingface.co/datasets/lmms-lab/ChartQA).

## Splits

| Split | Rows |
|-------|------|
| `test.lance` | 2,500 |

> The `lmms-lab/ChartQA` redistribution exposes test only. Train and validation live in the original release (https://github.com/vis-nlp/ChartQA); add them via `chartqa/dataprep.py --splits` once a parquet mirror is identified.

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | `int64` | Row index |
| `image` | `large_binary` | Inline chart image bytes |
| `image_id` / `question_id` | `string?` | (Source does not assign explicit ids — null for now) |
| `question` | `string` | Natural-language question |
| `answers` | `list<string>` | Reference answer (typically a single string) |
| `answer` | `string` | First answer — used as canonical |
| `type` | `string?` | Question type (`human` vs `augmented`) |
| `image_emb` | `fixed_size_list<float32, 512>` | CLIP image embedding (cosine-normalized) |
| `question_emb` | `fixed_size_list<float32, 512>` | CLIP text embedding of the question |

## Pre-built indices

- `IVF_PQ` on `image_emb` and `question_emb` — `metric=cosine`
- `INVERTED` (FTS) on `question` and `answer`
- `BITMAP` on `type`

## Quick start

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/chartqa-lance/data/test.lance")
print(ds.count_rows(), ds.schema.names, ds.list_indices())
```

## Source & license

Converted from [`lmms-lab/ChartQA`](https://huggingface.co/datasets/lmms-lab/ChartQA). The original ChartQA dataset is released under the GNU GPL-3.0 license by Masry et al.

## Citation

```
@inproceedings{masry2022chartqa,
  title={ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
  author={Masry, Ahmed and Long, Do Xuan and Tan, Jia Qing and Joty, Shafiq and Hoque, Enamul},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  year={2022}
}
```
