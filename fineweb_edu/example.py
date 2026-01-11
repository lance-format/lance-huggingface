#!/usr/bin/env python3
"""FineWeb-Edu Lance Dataset - Streaming & Lance usage examples."""

import textwrap

import datasets
import lance
import pyarrow as pa


# ============================================================================
# 1. Load FineWeb-Edu via Hugging Face streaming & grab a batch
# ============================================================================

def load_using_hf():
    ds = datasets.load_dataset(
        "lance-format/fineweb-edu",
        split="train",
        streaming=True,
    )
    print(ds)
    return ds


def get_hf_stream_batch(ds, batch_size=5):
    """Consume a batch from the Hugging Face IterableDataset."""
    rows = list(ds.take(batch_size))
    return pa.Table.from_pylist(rows)


# ============================================================================
# 2. Load Lance dataset directly & showcase queries
# ============================================================================

def load_dataset():
    ds = lance.dataset("hf://datasets/lance-format/fineweb-edu")
    print(f"✓ Loaded {ds.count_rows():,} passages")
    return ds


def _existing_columns(ds, preferred):
    schema_fields = set(ds.schema.names)
    cols = [col for col in preferred if col in schema_fields]
    if not cols:
        cols = ds.schema.names[: len(preferred)]
    return cols


def show_samples(ds, limit=5, offset=0):
    columns = _existing_columns(ds, ["title", "language", "text"])
    scanner = ds.scanner(
        columns=columns,
        limit=limit,
        offset=offset,
    )
    docs = scanner.to_table().to_pylist()

    print(f"\nSample passages (rows {offset}-{offset + len(docs)}):")
    for i, doc in enumerate(docs, start=offset):
        title = doc.get("title") or doc.get("url") or "(no title)"
        lang = doc.get("language") or "?"
        text = (doc.get("text") or "").replace("\n", " ")
        preview = textwrap.shorten(text, width=160, placeholder="...")
        print(f"[{i}] {title} [{lang}] -> {preview}")


def vector_search(ds, query_embedding, embedding_type, k=5, nprobes=1, refine_factor=1):
    if "text_embedding" not in ds.schema.names:
        raise RuntimeError("Dataset has no 'text_embedding' column—run embeddings backfill first.")
    query = pa.array([query_embedding], type=embedding_type)
    return ds.scanner(
        nearest={
            "column": "text_embedding",
            "q": query[0],
            "k": k,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
        },
        columns=["text", "text_embedding"],
    ).to_table().to_pylist()


def find_similar_passages(ds, doc_index=0, k=5, nprobes=1):
    if "text_embedding" not in ds.schema.names:
        print("Skipping vector search; 'text_embedding' column not found.")
        return
    ref_table = ds.take([doc_index], columns=["text", "text_embedding"])
    ref = ref_table.to_pylist()[0]
    emb_type = ref_table.schema.field("text_embedding").type

    results = vector_search(
        ds,
        ref["text_embedding"],
        emb_type,
        k=k + 1,
        nprobes=nprobes,
        refine_factor=max(10, k * 2),
    )

    print(f"\nQuery passage [{doc_index}]:")
    print(textwrap.shorten((ref.get("text") or "").replace("\n", " "), width=200))

    print(f"\nTop {k} similar passages:")
    for rank, doc in enumerate(results[1:k + 1], start=1):
        preview = textwrap.shorten((doc.get("text") or "").replace("\n", " "), width=120)
        print(f"  {rank}. {preview}")


if __name__ == "__main__":
    print("\nLoading FineWeb-Edu via Hugging Face streaming...")
    hf_ds = load_using_hf()
    batch = get_hf_stream_batch(hf_ds)
    print(batch)

    print("\nLoading Lance dataset for interactive queries...")
    print(
        "NOTE: The Hub copy is huge; vector/FTS demos require local Lance storage. "
        "Pre-built indices are coming soon—run these ops after downloading."
    )
    ds = load_dataset()

    show_samples(ds, limit=3, offset=0)
    find_similar_passages(ds, doc_index=42, k=5, nprobes=8)
