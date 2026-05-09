---
license: cc-by-sa-3.0
task_categories:
- question-answering
- text-retrieval
language:
- en
tags:
- natural-questions
- open-domain-qa
- multi-hop-qa
- lance
- sentence-transformers
pretty_name: natural-questions-val-lance
size_categories:
- 1K<n<10K
---
# Natural Questions — Validation (Lance Format)

Lance-formatted version of the [Natural Questions](https://ai.google.com/research/NaturalQuestions/) **validation split** — 7,830 real Google search queries with their full Wikipedia articles and 1–5 annotator labels per question. Sourced from [`google-research-datasets/natural_questions`](https://huggingface.co/datasets/google-research-datasets/natural_questions).

> The NQ **train** split is 143 GB (307,373 rows); it is intentionally not bundled here. Add it via `natural_questions/dataprep.py --splits train` once disk + bandwidth allow.

## Splits

| Split | Rows |
|-------|------|
| `validation.lance` | 7,830 |

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | `string` | NQ example id |
| `question` | `string` | Original Google search query |
| `document_title` | `string` | Wikipedia article title |
| `document_url` | `string` | Wikipedia article URL |
| `document_html` | `large_binary` | Full HTML of the article (inline; UTF-8 bytes) |
| `short_answers` | `list<string>` | Deduped short-answer spans across all annotators |
| `num_short_answers` | `int32` | Total annotator spans (incl. duplicates) |
| `has_short_answer` | `bool` | At least one annotator provided a short-answer span |
| `has_long_answer` | `bool` | At least one annotator selected a long-answer candidate |
| `yes_no_answer` | `string` | `YES` / `NO` / `NONE` — majority vote across annotators |
| `question_emb` | `fixed_size_list<float32, 384>` | sentence-transformers `all-MiniLM-L6-v2` (cosine-normalized) |

## Pre-built indices

- `IVF_PQ` on `question_emb` — `metric=cosine`
- `INVERTED` (FTS) on `question`
- `BTREE` on `id`, `document_title`
- `BITMAP` on `yes_no_answer`, `has_short_answer`, `has_long_answer`

## Quick start

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/natural-questions-val-lance/data/validation.lance")
print(ds.count_rows(), ds.schema.names, ds.list_indices())
```

## Get only questions with short-answer spans

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/natural-questions-val-lance/data/validation.lance")
short = ds.scanner(
    filter="has_short_answer = true",
    columns=["question", "short_answers", "document_title"],
    limit=10,
).to_table().to_pylist()
```

## Read the full Wikipedia HTML for one question

```python
import lance
ds = lance.dataset("hf://datasets/lance-format/natural-questions-val-lance/data/validation.lance")
row = ds.take([0], columns=["question", "document_html", "document_url"]).to_pylist()[0]
print(row["question"], "->", row["document_url"])
print(row["document_html"][:500].decode("utf-8", errors="replace"))
```

## Source & license

Converted from [`google-research-datasets/natural_questions`](https://huggingface.co/datasets/google-research-datasets/natural_questions). NQ is released under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) (matching the Wikipedia source).

## Citation

```
@article{kwiatkowski2019natural,
  title={Natural Questions: A Benchmark for Question Answering Research},
  author={Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and Toutanova, Kristina and Jones, Llion and Kelcey, Matthew and Chang, Ming-Wei and Dai, Andrew M. and Uszkoreit, Jakob and Le, Quoc and Petrov, Slav},
  journal={Transactions of the Association for Computational Linguistics},
  year={2019}
}
```
