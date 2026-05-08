# Tracked Datasets

All datasets converted to Lance format under [`huggingface.co/lance-format`](https://huggingface.co/lance-format). Every row of every dataset stores its multimodal data **inline** (image / audio / video bytes + embeddings + metadata) — no sidecar files. Pre-built `IVF_PQ` / FTS / scalar indices ship inside each Lance directory.

| # | Folder | HF repo | Modality / task | Source | Splits & rows | Embeddings & indices | Status |
|---|---|---|---|---|---|---|---|
| 1 | `mnist/` | [`mnist-lance`](https://huggingface.co/datasets/lance-format/mnist-lance) | Image classification | `ylecun/mnist` | train 60,000 · test 10,000 | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| 2 | `cifar10/` | [`cifar10-lance`](https://huggingface.co/datasets/lance-format/cifar10-lance) | Image classification | `uoft-cs/cifar10` | train 50,000 · test 10,000 | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| 3 | `fashion_mnist/` | [`fashion-mnist-lance`](https://huggingface.co/datasets/lance-format/fashion-mnist-lance) | Image classification | `zalando-datasets/fashion_mnist` | train 60,000 · test 10,000 | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| 4 | `imagenet1k_val/` | [`imagenet-1k-val-lance`](https://huggingface.co/datasets/lance-format/imagenet-1k-val-lance) | Image classification | `benjamin-paine/imagenet-1k` (val) | validation 50,000 | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| 5 | `eurosat/` | [`eurosat-lance`](https://huggingface.co/datasets/lance-format/eurosat-lance) | Geo / satellite tile classification | `blanchon/EuroSAT_RGB` | train 16,200 · val 5,400 · test 5,400 | CLIP IVF_PQ + BTREE/BITMAP | ✅ |
| 6 | `pascal_voc_2012/` | [`pascal-voc-2012-segmentation-lance`](https://huggingface.co/datasets/lance-format/pascal-voc-2012-segmentation-lance) | Semantic segmentation | `nateraw/pascal-voc-2012` | train 1,464 · validation 1,449 | CLIP IVF_PQ | ✅ |
| 7 | `ade20k/` | [`ade20k-lance`](https://huggingface.co/datasets/lance-format/ade20k-lance) | Scene parsing (semantic + instance segmentation) | `1aurent/ADE20K` | train 25,574 · validation 2,000 | CLIP IVF_PQ + BTREE + LABEL_LIST on `objects_present` | ✅ |
| 8 | `coco_detection_2017/` | [`coco-detection-2017-lance`](https://huggingface.co/datasets/lance-format/coco-detection-2017-lance) | Object detection (boxes + classes + areas) | `detection-datasets/coco` | train 117,266 · val 4,952 | CLIP IVF_PQ + BTREE + LABEL_LIST on `categories_present` | ✅ |
| 9 | `kitti/` | [`kitti-2d-detection-lance`](https://huggingface.co/datasets/lance-format/kitti-2d-detection-lance) | Autonomous-driving 2D + 3D detection | `nateraw/kitti` | train 7,481 | CLIP IVF_PQ + BTREE + LABEL_LIST on `types_present` | ✅ |
| 10 | `flickr30k/` | [`flickr30k-lance`](https://huggingface.co/datasets/lance-format/flickr30k-lance) | Image-caption | `lmms-lab/flickr30k` | train 31,783 | CLIP image+text IVF_PQ + FTS on caption | ✅ |
| 11 | `coco_captions_2017/` | [`coco-captions-2017-lance`](https://huggingface.co/datasets/lance-format/coco-captions-2017-lance) | Image-caption | `lmms-lab/COCO-Caption2017` | val 5,000 · test 40,670 | CLIP image+text IVF_PQ + FTS on caption | ✅ |
| 12 | `vqav2/` | [`vqav2-lance`](https://huggingface.co/datasets/lance-format/vqav2-lance) | Visual question answering | `lmms-lab/VQAv2` | validation 214,354 | CLIP image emb + CLIP text emb on question · IVF_PQ on both · FTS on question · BTREE on ids · BITMAP on answer/question type | ✅ (val only — train deferred, see card) |
| 13 | `squad_v2/` | [`squad-v2-lance`](https://huggingface.co/datasets/lance-format/squad-v2-lance) | Question answering | `rajpurkar/squad_v2` | train 130,319 · validation 11,873 | MiniLM IVF_PQ + dual FTS + BTREE/BITMAP | ✅ |
| 14 | `triviaqa/` | [`trivia-qa-lance`](https://huggingface.co/datasets/lance-format/trivia-qa-lance) | Question answering | `mandarjoshi/trivia_qa` (rc.nocontext) | train 138,384 · validation 17,944 | MiniLM IVF_PQ + FTS + BTREE/BITMAP | ✅ |
| 15 | `ms_marco/` | [`ms-marco-v2.1-lance`](https://huggingface.co/datasets/lance-format/ms-marco-v2.1-lance) | Passage retrieval / IR | `microsoft/ms_marco` (v2.1) | train 808,731 · validation 101,093 | MiniLM IVF_PQ + dual FTS + BTREE/BITMAP | ✅ |
| 16 | `librispeech/` | [`librispeech-clean-lance`](https://huggingface.co/datasets/lance-format/librispeech-clean-lance) | ASR (audio + transcript) | `openslr/librispeech_asr` (clean) | train.100 28,539 · dev 2,703 · test 2,620 | MiniLM transcript IVF_PQ + FTS on transcript + BTREE on ids | ✅ |
| 17 | `lerobot/xvla-soft-fold/` | [`lerobot-xvla-soft-fold`](https://huggingface.co/datasets/lance-format/lerobot-xvla-soft-fold) | Robotics episodes (LeRobot v3.0) | LeRobot xVLA soft-fold | frames + videos + episodes | per-camera blob + episode segments | ✅ (existing) |
| 18 | `lerobot/pusht/` | [`lerobot-pusht-lance`](https://huggingface.co/datasets/lance-format/lerobot-pusht-lance) | Robotics episodes (Diffusion Policy PushT) | `lerobot/pusht` | frames 25,650 · videos 1 · episodes 206 | per-camera blob + episode segments | ✅ |
| 19 | `laion-1M/` | [`laion-1m`](https://huggingface.co/datasets/lance-format/laion-1m) | Image-caption (LAION subset) | LAION-5B subset | train 1,160,000 | CLIP IVF_PQ + FTS on caption | ✅ (existing) |
| 20 | `openvid_hf/` | [`openvid-lance`](https://huggingface.co/datasets/lance-format/openvid-lance) | Text-to-video | `nkp37/OpenVid-1M` | train 937,957 | 1024-d video emb · IVF_PQ · FTS on caption · video blobs | ✅ (existing) |
| 21 | `fineweb_edu/` & `fineweb/` | [`fineweb-edu`](https://huggingface.co/datasets/lance-format/fineweb-edu) | Text corpus (web pre-training) | `HuggingFaceFW/fineweb-edu` | train ≈ 1.53 B | Cohere text emb + IVF_PQ + FTS | ✅ (existing) |

Status legend: ✅ live · ⏳ in flight · ⛔ skipped (see notes).

## Skipped (with reason)

| Dataset | Reason |
|---|---|
| Cityscapes | Requires registration with cityscapes-dataset.com; no anonymous download. |
| Waymo Open | >1 TB; out of disk budget for this batch. |
| Argoverse 2 | Multi-hundred-GB sensor logs; out of disk budget. |
| FineVision | Per maintainer guidance — out of scope. |
| ImageNet-1k train | 1.28 M images / ~155 GB JPEG; only the 50k validation split is bundled. |
| nuScenes | Requires registration; consider mini split (~4 GB) in a follow-up. |
| Open Images v7 | ~9 M images / ~500 GB; convert as a sampled subset in a future batch. |

## Conventions

- Image / mask / audio / video bytes stored **inline** as `large_binary` columns. Lance blob encoding is reserved for multi-MB-per-row media (videos in `openvid` / `lerobot/*`).
- Image embeddings: OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` (512-d, L2-normalized).
- Text embeddings: sentence-transformers `all-MiniLM-L6-v2` (384-d, L2-normalized).
- IVF_PQ partitions sized to `sqrt(num_rows)` clamped to {16, 64, 256, 512, 1024}, sub-vectors at `dim/8`. Below 256 rows the vector index is skipped automatically (`_common/indexing.py`).
- FTS uses `with_position=False`, `remove_stop_words=False` (per [docs.lancedb.com/performance](https://docs.lancedb.com/performance)).
- `BITMAP` for low-cardinality categoricals (≤ ~1000 distinct), `LABEL_LIST` for `list<T>` filters, `BTREE` everywhere else.
- All conversions write with `max_bytes_per_file = 8 GiB`.
- Pushing is via `hf upload-large-folder`, serialized through a `flock` so multiple converters do not stomp on each other.

## How to add a new dataset

1. Create `<name>/dataprep.py` and `<name>/HF_DATASET_CARD.md`. Follow the patterns in `mnist/`, `flickr30k/`, `squad_v2/`, `librispeech/` (image cls, image-caption, text QA, audio respectively).
2. Reuse helpers from `_common/`:
   - `embeddings.CLIPEncoder` / `embeddings.SentenceEncoder` for image / text embeddings.
   - `indexing.build_default_indices(...)` for IVF_PQ + FTS + BTREE + BITMAP + LABEL_LIST.
   - `schemas.fixed_size_emb_field`, `schemas.blob_field` for column definitions.
   - `upload.push_to_hub` or the shell wrapper `_common/upload_and_cleanup.sh` for the Hub push.
3. Add the dataset to `_common/run_all.sh`'s queue (or run it manually).
4. Add a row in this table.
