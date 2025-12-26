# Fineweb Dataset with Embeddings

## Requirements

* `uv`
* `git`, `git-lfs` and [`git-xet`](https://huggingface.co/docs/hub/en/xet/using-xet-storage#git-xet)

## Prepare datasets

```sh
mkdir data && cd data
uvx hf download --repo-type dataset Cohere/fineweb-edu-emb --local-dir fineweb-edu-emb
uvx hf download --repo-type dataset Cohere/fineweb-edu-corpus --local-dir fineweb-edu-corpus
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