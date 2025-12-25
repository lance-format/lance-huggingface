# Fineweb Dataset with Embeddings

## Requirements

* `uv`
* `git`, `git-lfs` and [`git-xet`](https://huggingface.co/docs/hub/en/xet/using-xet-storage#git-xet)

## Prepare datasets

```
mkdir data && cd data
git clone --depth 1 https://huggingface.co/datasets/Cohere/fineweb-edu-emb
git clone --depth 1 https://huggingface.co/datasets/Cohere/fineweb-edu-corpus
```

```py
uv run dataprep.py
```