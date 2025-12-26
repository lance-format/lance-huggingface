---
tags:
- fineweb
- cohere
- lance
- embeddings
license: odc-by
task_categories:
  - text-generation
language:
  - en
pretty_name: Fineweb-Edu Cohere Embeddings
size_categories:
- 100M<n<1B
source_datasets:
- Cohere/fineweb-edu-emb
- Cohere/fineweb-edu-corpus
---

# Fineweb Edu Lance dataset with Cohere embeddings

This dataset is [Fineweb Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
with Cohere `embed-multilingual-v3.0` embeddings, written in [Lance format](https://lance.org/).

## How to use

It works with huggingface `datasets` package

```py
import datasets
ds = datasets.load_dataset("lance-format/fineweb-edu-cohere-emb")
```

To take full advantages of Lance format, you can also download the dataset via `git`

```sh
export HF_TOKEN=hf_....
uvx hf download --repo-type dataset lance-format/fineweb-edu-cohere-emb --local-dir fineweb-edu
```

Run in `python`

```py
import lance

ds = lance.dataset("./fineweb-edu/data/train.lance")
print(ds.count_rows())
```
