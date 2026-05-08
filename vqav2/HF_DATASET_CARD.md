---
license: cc-by-4.0
task_categories:
- visual-question-answering
- image-text-to-text
- image-feature-extraction
language:
- en
tags:
- vqa
- vqav2
- vision-language
- lance
- clip-embeddings
pretty_name: vqav2-lance
size_categories:
- 100K<n<1M
---
# VQAv2 (Lance Format)

Lance-formatted version of [VQAv2](https://visualqa.org/) — Visual Question Answering on COCO images, sourced from [`lmms-lab/VQAv2`](https://huggingface.co/datasets/lmms-lab/VQAv2). Each row is a `(image, question, 10 answers)` triple with **two** CLIP embeddings (image + question text) so the same dataset supports both visual retrieval and question-similarity retrieval.

## Splits

| Split | Rows |
|-------|------|
| `validation.lance` | 214,354 |
| `train.lance`      | 443,757 |

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | `int64` | Row index within split |
| `image` | `large_binary` | Inline JPEG bytes |
| `image_id` | `int64` | COCO image id |
| `question_id` | `int64` | VQAv2 question id |
| `question` | `string` | Natural-language question |
| `question_type` | `string` | First few tokens of the question (e.g. `what is`, `is the`) |
| `answer_type` | `string` | One of `yes/no`, `number`, `other` |
| `multiple_choice_answer` | `string` | Canonical (most-common) answer |
| `answers` | `list<string>` | Raw answers from 10 annotators |
| `answer_confidences` | `list<string>` | Parallel confidence list (`yes` / `maybe` / `no`) |
| `image_emb` | `fixed_size_list<float32, 512>` | OpenCLIP `ViT-B-32` image embedding (cosine-normalized) |
| `question_emb` | `fixed_size_list<float32, 512>` | OpenCLIP `ViT-B-32` text embedding of the question (cosine-normalized) |

Because both embeddings come from the same CLIP model, they share an embedding space and cross-modal retrieval (image→question or question→image) works out of the box.

## Pre-built indices

- `IVF_PQ` on `image_emb` and `question_emb` — `metric=cosine`
- `INVERTED` (FTS) on `question`
- `BTREE` on `image_id`, `question_id`, `multiple_choice_answer`
- `BITMAP` on `question_type`, `answer_type`

## Quick start

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/vqav2-lance/data/validation.lance")
print(ds.count_rows(), ds.schema.names, ds.list_indices())
```

## Cross-modal: find an image for a free-form question

```python
import lance
import pyarrow as pa
import open_clip
import torch

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.eval().cuda().half()
with torch.no_grad():
    q = model.encode_text(tokenizer(["what color is the dog?"]).cuda())
    q = (q / q.norm(dim=-1, keepdim=True)).float().cpu().numpy()[0]

ds = lance.dataset("hf://datasets/lance-format/vqav2-lance/data/validation.lance")
emb_field = ds.schema.field("image_emb")
hits = ds.scanner(
    nearest={"column": "image_emb", "q": pa.array([q.tolist()], type=emb_field.type)[0], "k": 5},
    columns=["image_id", "question", "multiple_choice_answer"],
).to_table().to_pylist()
```

## Question similarity (text→text)

```python
ds = lance.dataset("hf://datasets/lance-format/vqav2-lance/data/validation.lance")
ref = ds.take([0], columns=["question_emb", "question"]).to_pylist()[0]
emb_field = ds.schema.field("question_emb")
neighbors = ds.scanner(
    nearest={"column": "question_emb", "q": pa.array([ref["question_emb"]], type=emb_field.type)[0], "k": 5},
    columns=["question", "multiple_choice_answer"],
).to_table().to_pylist()
print("query:", ref["question"])
for n in neighbors:
    print(n)
```

## Filter by question / answer type

```python
ds = lance.dataset("hf://datasets/lance-format/vqav2-lance/data/validation.lance")
yesno = ds.scanner(filter="answer_type = 'yes/no'", columns=["question", "multiple_choice_answer"], limit=5).to_table()
counts = ds.scanner(filter="answer_type = 'number'", columns=["question", "multiple_choice_answer"], limit=5).to_table()
```

## Why Lance?

- One dataset for images + questions + answers + dual embeddings + indices — no JSON/CSV sidecars.
- On-disk vector and FTS indices live next to the data, so search works on local copies and on the Hub.
- Schema evolution: add columns (alternate embeddings, model predictions, generated answers) without rewriting the data.

## Source & license

Converted from [`lmms-lab/VQAv2`](https://huggingface.co/datasets/lmms-lab/VQAv2). VQAv2 questions and annotations are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying images come from COCO and are subject to Flickr terms of service. See the [VQAv2 download page](https://visualqa.org/download.html) for details.

## Citation

```
@inproceedings{goyal2017making,
  title={Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering},
  author={Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
