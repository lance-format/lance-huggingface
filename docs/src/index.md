# Lance-Hugging Face Integration

Lance brings the power of a modern, columnar lakehouse format to the [Hugging Face Hub](https://huggingface.co/). This integration lets you scan, filter, and search multimodal datasets (stored in Lance tables) directly on the Hub using the `hf://` URL scheme, without pulling entire datasets down to local disk.

## Quickstart

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

## Examples

Head to the [examples](examples.md) page for installation, authentication, and runnable snippets.

## Why Lance?

When you persist your data in Lance format on the Hugging Face Hub, you get all of the following benefits:

1. **Optimized for ML and AI**: Lance is a modern columnar format designed for fast random access without compromising scan performance.
1. **Multimodal data support**: Binary objects (blobs), embeddings, and traditional scalar data all live in one place, as one tabular dataset -- this makes it easier to govern, share, and distribute it via the Hub.
1. **Vector, FTS and secondary indexes** are first-class citizens in the format. Lance comes with fast, on-disk, scalable vector and FTS indexes that sit right alongside the dataset on the Hub, so you can share not only your data but also your embeddings and indexes without your users needing to recompute them.first-class citizens, and native to the format itself. 
1. **Flexible schema**: Add new features/columns (moderation tags, embeddings, etc.) **without** needing to rewrite the entire table.

!!! tip "Scan, search and share your large datasets with ease"
    Because multimodal assets and indexes and are first-class citizens in Lance, you can store your scalar data, blobs, embeddings **and** indexes together in **one** dataset on the Hub, simplifying sharing and distribution. You can also run filtered search and vector search queries without needing to copy the entire dataset locally first.
    
    If the creator of a Lance dataset on the Hugging Face Hub put in the work to compute embeddings and an index, you can immediately benefit from their work, too!


## Explore more Lance datasets

Lance is an open lakehouse format with native support for multimodal blobs alongside your traditional tabular data. You can work with images, audio, video, text, embeddings, and scalar metadata all in one place.

Explore more Lance datasets on the [Hugging Face Hub](https://huggingface.co/datasets?search=lance-format&sort=downloads). Feel free to upload and share your own Lance datasets too!