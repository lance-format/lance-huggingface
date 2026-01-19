# Lance-Hugging Face Integration

Lance brings the power of a modern, columnar lakehouse format to the [Hugging Face Hub](https://huggingface.co/). This integration lets you scan, filter, and search multimodal datasets (stored in Lance tables) directly on the Hub using the `hf://` URL scheme, without pulling entire datasets down to local disk.

## Quickstart

Simply point to a Lance dataset directory to scan the dataset. You can then pass in projections and filters to the scanner object, and limit the results as required to avoid large I/O.

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

scanner = ds.scanner(
    columns=["caption", "url", "similarity"],
    limit=5
)

rows = scanner.to_table().to_pylist()
for row in rows:
    print(row)
```


## Why Lance?

When you persist your data in Lance format on the Hugging Face Hub, you get all of the following benefits:

1. **Optimized for ML and AI**: Lance is a modern columnar format designed for fast random access without compromising scan performance.
1. **Multimodal data support**: Binary objects (blobs), embeddings, and traditional scalar data all live in one place, as one tabular dataset -- this makes it easier to govern, share, and distribute it via the Hub.
1. **Vector, FTS and secondary indexes** are first-class citizens in the format. Lance comes with fast, on-disk, scalable vector and FTS indexes that sit right alongside the dataset on the Hub, so you can share not only your data but also your embeddings and indexes without your users needing to recompute them.first-class citizens, and native to the format itself. 
1. **Flexible schema**: Add new features/columns (moderation tags, embeddings, etc.) **without** needing to rewrite the entire table.

!!! tip "Scan, search and share your large datasets with ease"
    Because multimodal assets and indexes and are first-class citizens in Lance, you can store your scalar data, blobs, embeddings **and** indexes together in **one** dataset on the Hub, simplifying sharing and distribution. You can also run filtered search and vector search queries without needing to copy the entire dataset locally first.
    
    If the creator of a Lance dataset on the Hugging Face Hub put in the work to compute embeddings and an index, you can immediately benefit from their work, too!

## Examples

This page collects practical examples you can use to run queries on Lance datasets on the Hugging Face Hub.
Installation, authentication, and runnable snippets are included below.

### Getting started

Install the core dependencies:

```bash
pip install pylance pyarrow
```

### Authentication

To read private datasets and get higher rate limits, export your Hugging Face token:

```shell
export HF_TOKEN="your_hugging_face_token_here"
```

!!! tip
    Using a Hugging Face token helps you avoid throttling and rate limiting when sending multiple requests to the Hub to query Lance datasets stored there.

### Scan a multimodal Lance dataset

You can open a Lance dataset stored on the Hub using the `hf://` path specifier. This scans the remote dataset without downloading it locally, and you can set limits, filters, and projections to only fetch the data you need.

```python
import lance

# Return as a Lance dataset
ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

scanner = ds.scanner(
    columns=["caption", "url", "similarity"],
    limit=5
)

rows = scanner.to_table().to_pylist()
for row in rows:
    print(row)
```

### Vector search on image embeddings

Because indexes are native to the Lance format, you can store embeddings and indexes together in one dataset and query them directly on the Hub.

If the dataset includes a vector index (for example `img_emb`), you can run vector search queries directly on the remote dataset without downloading it. The example below shows a nearest neighbor search using an image embedding as the query vector.

```python
import lance
import pyarrow as pa

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

emb_field = ds.schema.field("img_emb")
ref = ds.take([0], columns=["img_emb"]).to_pylist()[0]["img_emb"]
query = pa.array([ref], type=emb_field.type)

neighbors = ds.scanner(
    nearest={
        "column": emb_field.name,
        "q": query[0],
        "k": 6,
        "nprobes": 16,
        "refine_factor": 30,
    },
    columns=["caption", "url", "similarity"],
).to_table().to_pylist()
```

### Work with video blobs

Large multimodal binary objects (blobs) are first-class citizens in Lance. The OpenVid-1M dataset from [this paper](https://arxiv.org/abs/2407.02371) is a good example; it contains high-quality videos and captions, with the video data stored in the `video_blob` column.

```python
import lance

lance_ds = lance.dataset("hf://datasets/lance-format/openvid-lance")
blob_file = lance_ds.take_blobs("video_blob", ids=[0])[0]
video_bytes = blob_file.read()
```

For large video blobs (for example, around 10 MB or beyond), Lance also provides a high-level [blob API](https://lance.org/guide/blob/) to store, distribute, and search them efficiently. The following example shows how to browse metadata without loading blobs, then fetch a blob on demand.

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/openvid-lance/data/train.lance")

# 1. Browse metadata without loading video blobs.
metadata = ds.scanner(
    columns=["caption", "aesthetic_score"],
    filter="aesthetic_score >= 4.5",
    limit=2,
).to_table().to_pylist()

# 2. Fetch a single video blob by row index.
selected_index = 0
blob_file = ds.take_blobs("video_blob", ids=[selected_index])[0]
with open("video_0.mp4", "wb") as f:
    f.write(blob_file.read())
```

## Explore more Lance datasets

Lance is an open lakehouse format with native support for multimodal blobs alongside your traditional tabular data -- so you can work with images, audio, video, text, embeddings, and scalar metadata all in one place.

This page gave a quick introduction to working with Lance datasets on the Hugging Face Hub. You can explore more datasets as we upload them to the [lance-format](https://huggingface.co/datasets?search=lance-format&sort=downloads) organization on the Hub. In the meantime, feel free to upload and share your own Lance datasets too!
