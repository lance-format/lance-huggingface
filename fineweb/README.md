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


## Upload to Huggingface

```sh
export HF_TOKEN=hf_....
uvx hf repo create lance-format/fineweb-edu-cohere-emb
uvx hf upload-large-folder lance-format/fineweb-edu-cohere-emb --repo-type=dataset fineweb-edu-cohere
```