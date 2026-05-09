# lance-huggingface
LanceDB conversions of standard ML datasets, published under the [`lance-format`](https://huggingface.co/lance-format) Hugging Face organization.


## Data organization

You can either include a `README.md` with [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards)
directly in a raw Lance dataset, or adopt a Hugging Face–style directory structure,
placing Lance datasets for different splits in the `/data` directory:

```sh
my_dataset/
├── README.md
└── data/
    ├── train.lance
    ├── test.lance
    └── validation.lance
```

## Datasets in this repo

Every folder is a self-contained converter — `dataprep.py` pulls the source from the Hub, writes a Lance dataset with inline multimodal data + indices, and uploads the bundle to `lance-format/<name>`. Cards in `HF_DATASET_CARD.md` follow the pattern of the existing converters (laion-1m, openvid).

| Folder | HF repo | Modality | Embeddings & indices |
|---|---|---|---|
| `mnist/`              | [`lance-format/mnist-lance`](https://huggingface.co/datasets/lance-format/mnist-lance)               | image classification | CLIP ViT-B/32 image emb · IVF_PQ · BTREE/BITMAP on label |
| `cifar10/`            | [`lance-format/cifar10-lance`](https://huggingface.co/datasets/lance-format/cifar10-lance)           | image classification | CLIP ViT-B/32 image emb · IVF_PQ · BTREE/BITMAP on label |
| `fashion_mnist/`      | [`lance-format/fashion-mnist-lance`](https://huggingface.co/datasets/lance-format/fashion-mnist-lance) | image classification | CLIP ViT-B/32 image emb · IVF_PQ · BTREE/BITMAP on label |
| `imagenet1k_val/`     | [`lance-format/imagenet-1k-val-lance`](https://huggingface.co/datasets/lance-format/imagenet-1k-val-lance) | image classification (50k val) | CLIP ViT-B/32 image emb · IVF_PQ |
| `pascal_voc_2012/`    | [`lance-format/pascal-voc-2012-segmentation-lance`](https://huggingface.co/datasets/lance-format/pascal-voc-2012-segmentation-lance) | semantic segmentation | CLIP ViT-B/32 image emb · IVF_PQ |
| `flickr30k/`          | [`lance-format/flickr30k-lance`](https://huggingface.co/datasets/lance-format/flickr30k-lance)       | image-caption | CLIP image+text · IVF_PQ on both · FTS on caption |
| `coco_captions_2017/` | [`lance-format/coco-captions-2017-lance`](https://huggingface.co/datasets/lance-format/coco-captions-2017-lance) | image-caption | CLIP image+text · IVF_PQ on both · FTS on caption |
| `squad_v2/`           | [`lance-format/squad-v2-lance`](https://huggingface.co/datasets/lance-format/squad-v2-lance)         | question-answering | MiniLM question emb · IVF_PQ · FTS on question/context |
| `triviaqa/`           | [`lance-format/trivia-qa-lance`](https://huggingface.co/datasets/lance-format/trivia-qa-lance)       | question-answering | MiniLM question emb · IVF_PQ · FTS on question |
| `ms_marco/`           | [`lance-format/ms-marco-v2.1-lance`](https://huggingface.co/datasets/lance-format/ms-marco-v2.1-lance) | passage retrieval | MiniLM query emb · IVF_PQ · FTS on query/selected_passage |
| `laion-1M/`           | [`lance-format/laion-1m`](https://huggingface.co/datasets/lance-format/laion-1m)                     | image-caption (1M) | CLIP image emb · IVF_PQ · FTS on caption |
| `openvid_hf/`         | [`lance-format/openvid-lance`](https://huggingface.co/datasets/lance-format/openvid-lance)           | text-to-video | video blobs · 1024-d emb · IVF_PQ · FTS on caption |
| `lerobot/xvla-soft-fold/` | [`lance-format/lerobot-xvla-soft-fold`](https://huggingface.co/datasets/lance-format/lerobot-xvla-soft-fold) | robotics episodes | episode video blobs |
| `fineweb/` & `fineweb_edu/` | [`lance-format/fineweb-edu`](https://huggingface.co/datasets/lance-format/fineweb-edu) | text corpus | Cohere embeddings · IVF_PQ · FTS |

## Shared helpers — `_common/`

- `embeddings.py` — GPU-batched OpenCLIP and sentence-transformer encoders (single-model load + L2-normalized output).
- `indexing.py` — `IVF_PQ` / FTS / BTREE / BITMAP builders with sensible defaults derived from row count and embedding dim, following [docs.lancedb.com/performance](https://docs.lancedb.com/performance).
- `schemas.py` — `pa.field` helpers for fixed-size embedding columns and blob-tagged fields.
- `upload.py` + `upload_and_cleanup.sh` — wrap `hf upload-large-folder`, then remove the local copy on success so disk frees up between datasets.
- `image_classification.py` and `image_caption.py` — generic write paths for the two most common image-dataset shapes.
- `run_all.sh` — orchestrator: runs conversions sequentially (single GPU) and overlaps uploads via a `flock`-serialized queue (HF documentation discourages parallel large-folder uploads).

## Performance / best practices

The shared helpers pin a few values from the [LanceDB performance guide](https://docs.lancedb.com/performance):

- Iterator ingest with several-thousand-row batches (`pa.RecordBatch.from_pydict`) — one commit per dataset, not per row.
- `max_bytes_per_file = 8 GiB` so individual Lance fragments stay manageable.
- `IVF_PQ` partitions sized by `sqrt(num_rows)` clamped to 16/64/256/512/1024 buckets; sub-vectors at `dim/8` (8-dim PQ chunks) by default.
- Embeddings are L2-normalized and indexed with `metric=cosine`.
- FTS index built with `with_position=False`, `remove_stop_words=False` to keep the index small.
- BITMAP for low-cardinality categorical columns (≤ ~1000 distinct values), BTREE everywhere else.
- All multimodal data — image bytes, masks, embeddings — stays inline in the Lance dataset, **not** as sidecar files.

## Running a conversion

```bash
# Convert a single dataset with embeddings and indices, then push to the Hub.
python mnist/dataprep.py --overwrite --push

# Or run several in sequence, with parallel uploads:
_common/run_all.sh mnist cifar10 fashion_mnist
```

Set `HF_TOKEN` (or run `hf auth login`) before pushing to the Hub.
