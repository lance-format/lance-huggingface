# Tracked Datasets

Catalog of datasets converted (or planned) under [`huggingface.co/lance-format`](https://huggingface.co/lance-format). Every entry has:

- a self-contained converter folder in this repo (`<name>/dataprep.py` + `HF_DATASET_CARD.md`),
- inline storage of every multimodal artefact (image bytes, masks, annotations, embeddings) — no sidecar files,
- pre-built indices (vector + FTS + scalar) where the row count and modality justify them,
- a Hub repo at `lance-format/<name>-lance` (or similar slug).

Status legend: ✅ live · 🟡 in flight · ⏳ queued · ⛔ skipped (with reason).

## Image classification

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `mnist/` | [`lance-format/mnist-lance`](https://huggingface.co/datasets/lance-format/mnist-lance) | ylecun/mnist | train · test | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| `cifar10/` | [`lance-format/cifar10-lance`](https://huggingface.co/datasets/lance-format/cifar10-lance) | uoft-cs/cifar10 | train · test | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| `fashion_mnist/` | [`lance-format/fashion-mnist-lance`](https://huggingface.co/datasets/lance-format/fashion-mnist-lance) | zalando-datasets/fashion_mnist | train · test | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| `imagenet1k_val/` | [`lance-format/imagenet-1k-val-lance`](https://huggingface.co/datasets/lance-format/imagenet-1k-val-lance) | benjamin-paine/imagenet-1k (val) | validation | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| `eurosat/` | [`lance-format/eurosat-lance`](https://huggingface.co/datasets/lance-format/eurosat-lance) | blanchon/EuroSAT_RGB | train · val · test | CLIP IVF_PQ + BTREE/BITMAP | ✅ |

## Object detection

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `coco_detection_2017/` | [`lance-format/coco-detection-2017-lance`](https://huggingface.co/datasets/lance-format/coco-detection-2017-lance) | detection-datasets/coco | train (117k) · val (5k) | CLIP IVF_PQ + BTREE + LABEL_LIST on `categories_present` | ✅ |
| `kitti/` | [`lance-format/kitti-2d-detection-lance`](https://huggingface.co/datasets/lance-format/kitti-2d-detection-lance) | nateraw/kitti | train (7,481) | CLIP IVF_PQ + BTREE + LABEL_LIST on `types_present` | ✅ |

## Semantic / instance segmentation

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `pascal_voc_2012/` | [`lance-format/pascal-voc-2012-segmentation-lance`](https://huggingface.co/datasets/lance-format/pascal-voc-2012-segmentation-lance) | nateraw/pascal-voc-2012 | train · validation | CLIP IVF_PQ | ✅ |
| `ade20k/` | [`lance-format/ade20k-lance`](https://huggingface.co/datasets/lance-format/ade20k-lance) | 1aurent/ADE20K | train (25,574) · validation (2,000) | CLIP IVF_PQ + BTREE + LABEL_LIST on `objects_present` | ✅ |

## Image captioning / vision-language

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `flickr30k/` | [`lance-format/flickr30k-lance`](https://huggingface.co/datasets/lance-format/flickr30k-lance) | lmms-lab/flickr30k | train (31.8k) | CLIP image+text IVF_PQ + FTS | ✅ |
| `coco_captions_2017/` | [`lance-format/coco-captions-2017-lance`](https://huggingface.co/datasets/lance-format/coco-captions-2017-lance) | lmms-lab/COCO-Caption2017 | val (5k) · test (40.7k) | CLIP image+text IVF_PQ + FTS | ✅ |

## Visual question answering

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `vqav2/` | [`lance-format/vqav2-lance`](https://huggingface.co/datasets/lance-format/vqav2-lance) | lmms-lab/VQAv2 | validation (214,354) | CLIP image emb + CLIP text emb on question · IVF_PQ on both · FTS on question · BTREE on ids · BITMAP on answer/question type | ✅ (val only — train deferred, see card) |

## Question answering / retrieval (text-only)

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `squad_v2/` | [`lance-format/squad-v2-lance`](https://huggingface.co/datasets/lance-format/squad-v2-lance) | rajpurkar/squad_v2 | train (130k) · validation (12k) | MiniLM IVF_PQ + dual FTS + BTREE/BITMAP | ✅ |
| `triviaqa/` | [`lance-format/trivia-qa-lance`](https://huggingface.co/datasets/lance-format/trivia-qa-lance) | mandarjoshi/trivia_qa (rc.nocontext) | train (138k) · validation (18k) | MiniLM IVF_PQ + FTS + BTREE/BITMAP | ✅ |
| `ms_marco/` | [`lance-format/ms-marco-v2.1-lance`](https://huggingface.co/datasets/lance-format/ms-marco-v2.1-lance) | microsoft/ms_marco (v2.1) | train (808k) · validation (101k) | MiniLM IVF_PQ + dual FTS + BTREE/BITMAP | ✅ |

## Robotics / world models

| Folder | Repo | Source | Splits | Indices | Status |
|---|---|---|---|---|---|
| `lerobot/xvla-soft-fold/` | [`lance-format/lerobot-xvla-soft-fold`](https://huggingface.co/datasets/lance-format/lerobot-xvla-soft-fold) | LeRobot v3.0 | frames · videos · episodes | per-camera blob + episode segments | ✅ (existing) |
| `lerobot/pusht/` | [`lance-format/lerobot-pusht-lance`](https://huggingface.co/datasets/lance-format/lerobot-pusht-lance) | lerobot/pusht | frames (25,650) · videos (1) · episodes (206) | per-camera blob + episode segments | ✅ |

## Already on the Hub (existing converters)

| Folder | Repo | Notes |
|---|---|---|
| `laion-1M/` | [`lance-format/laion-1m`](https://huggingface.co/datasets/lance-format/laion-1m) | 1M LAION images + CLIP emb + IVF_PQ |
| `openvid_hf/` | [`lance-format/openvid-lance`](https://huggingface.co/datasets/lance-format/openvid-lance) | 938k videos as blobs + 1024-dim emb + IVF_PQ + FTS |
| `fineweb/` & `fineweb_edu/` | [`lance-format/fineweb-edu`](https://huggingface.co/datasets/lance-format/fineweb-edu) | 1.5B-row text corpus with Cohere embeddings |

## Skipped (with reason)

| Dataset | Reason |
|---|---|
| Cityscapes | Requires registration with cityscapes-dataset.com (no anonymous access). Track via `cityscapes/` placeholder once a re-distribution path is identified. |
| Waymo Open Dataset | >1 TB; out of disk budget for this batch. |
| Argoverse 2 | Multi-hundred-GB sensor logs; out of disk budget. |
| FineVision | Per maintainer guidance — out of scope. |
| ImageNet-1k train | 1.28M images / ~155 GB JPEG; only the 50k validation split is included. Train conversion requires its own dedicated upload window. |
| nuScenes | Requires registration; consider mini split (~4 GB) in a follow-up batch. |
| Open Images v7 | ~9M images / ~500 GB; convert as a sampled subset in a future batch. |

## Conventions every dataset follows

- Image / mask / video bytes stored **inline** as `large_binary` columns (legacy blob encoding only for multi-MB rows like videos — see `_common/schemas.py`).
- Embedding columns are `fixed_size_list<float32, dim>`, L2-normalized, CLIP `ViT-B-32` (512-d) for images / image-text, MiniLM `all-MiniLM-L6-v2` (384-d) for text-only.
- IVF_PQ partitions sized to `sqrt(num_rows)` clamped to {16, 64, 256, 512, 1024}, sub-vectors at `dim/8`. Below 256 rows the vector index is skipped automatically (`_common/indexing.py`).
- FTS indices use `with_position=False`, `remove_stop_words=False` (per [docs.lancedb.com/performance](https://docs.lancedb.com/performance)).
- `BITMAP` on low-cardinality categorical columns (≤ ~1000 distinct values), `BTREE` everywhere else.
- All conversions write with `max_bytes_per_file = 8 GiB`.
- Converters emit a `data/<split>.lance` directory layout so the Hub viewer matches `datasets.load_dataset` semantics; the dataset card lives at the repo root as `README.md`.
- Pushing is via `hf upload-large-folder`, serialized through a `flock` so multiple converters do not stomp on each other.

## How to add a new dataset

1. Create `<name>/dataprep.py` and `<name>/HF_DATASET_CARD.md`. Follow the patterns in `mnist/`, `flickr30k/`, `squad_v2/` (image classification, image-caption, text QA respectively).
2. Reuse helpers from `_common/`:
   - `embeddings.CLIPEncoder` / `embeddings.SentenceEncoder` for image / text embeddings.
   - `indexing.build_default_indices(...)` for IVF_PQ + FTS + BTREE + BITMAP.
   - `schemas.fixed_size_emb_field`, `schemas.blob_field` for column definitions.
   - `upload.push_to_hub` or the shell wrapper `_common/upload_and_cleanup.sh` for the Hub push.
3. Add the dataset to `_common/run_all.sh`'s queue (or run it manually).
4. Append a new row in this file under the relevant section.
