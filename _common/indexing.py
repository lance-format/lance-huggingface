"""Index-building helpers with sensible defaults.

Following docs.lancedb.com/performance:
- IVF_PQ for general-purpose vector search (good below 256 dims; still useful for 384/512).
- BTREE for numeric/string scalar columns used in filters.
- BITMAP for low-cardinality categorical columns (<= ~1000 distinct values).
- FTS without ``with_position`` and without ``remove_stop_words`` unless we explicitly need them.

We pick ``num_partitions`` from the row count using a sqrt heuristic clamped to
sane bounds. ``num_sub_vectors`` is derived from the embedding dim so that
each sub-vector covers 8 dims (the textbook PQ rule of thumb).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


def _pick_num_partitions(num_rows: int) -> int:
    if num_rows < 10_000:
        return 16
    if num_rows < 100_000:
        return 64
    if num_rows < 500_000:
        return 256
    if num_rows < 2_000_000:
        return 512
    return min(4096, max(1024, int(math.sqrt(num_rows))))


def _pick_num_sub_vectors(dim: int) -> int:
    # 8 dims per sub-vector tends to be a good balance between recall and size.
    candidates = [d for d in (dim // 4, dim // 8, dim // 16) if d > 0 and dim % d == 0]
    return candidates[1] if len(candidates) >= 2 else (candidates[0] if candidates else 8)


def create_vector_index(
    dataset,
    column: str,
    *,
    dim: Optional[int] = None,
    metric: str = "cosine",
    num_partitions: Optional[int] = None,
    num_sub_vectors: Optional[int] = None,
    replace: bool = True,
) -> None:
    """Create an IVF_PQ index on ``column`` using sensible defaults.

    Skips silently if the dataset is too small to train PQ (< 256 rows or
    fewer rows than partitions).
    """
    rows = dataset.count_rows()
    if rows < 256:
        print(f"  skipping vector index on {column}: only {rows} rows (need >= 256 to train PQ)")
        return

    if dim is None:
        field = dataset.schema.field(column)
        list_size = getattr(field.type, "list_size", None)
        if list_size is None:
            raise ValueError(f"Column {column!r} is not a fixed-size list")
        dim = int(list_size)

    num_partitions = num_partitions or _pick_num_partitions(rows)
    # Lance requires at least one row per partition for kmeans training.
    num_partitions = max(1, min(num_partitions, max(1, rows // 64)))
    num_sub_vectors = num_sub_vectors or _pick_num_sub_vectors(dim)

    print(
        f"  vector index on {column} ({rows:,} rows, dim={dim}) "
        f"-> IVF_PQ partitions={num_partitions} sub_vectors={num_sub_vectors} metric={metric}"
    )
    dataset.create_index(
        column=column,
        index_type="IVF_PQ",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        metric=metric,
        replace=replace,
    )


def create_fts(dataset, column: str, *, replace: bool = True) -> None:
    """FTS index on a string column. Phrase search and stop-word filtering are
    disabled per the performance guide.
    """
    print(f"  FTS index on {column}")
    dataset.create_scalar_index(
        column,
        index_type="INVERTED",
        with_position=False,
        remove_stop_words=False,
        replace=replace,
    )


def create_btree(dataset, column: str, *, replace: bool = True) -> None:
    print(f"  BTREE index on {column}")
    dataset.create_scalar_index(column, index_type="BTREE", replace=replace)


def create_bitmap(dataset, column: str, *, replace: bool = True) -> None:
    print(f"  BITMAP index on {column}")
    dataset.create_scalar_index(column, index_type="BITMAP", replace=replace)


def build_default_indices(
    dataset,
    *,
    vector_columns: Iterable[str] = (),
    fts_columns: Iterable[str] = (),
    btree_columns: Iterable[str] = (),
    bitmap_columns: Iterable[str] = (),
    metric: str = "cosine",
) -> None:
    """One-shot helper to build all indices for a freshly written dataset."""
    for col in vector_columns:
        create_vector_index(dataset, col, metric=metric)
    for col in fts_columns:
        create_fts(dataset, col)
    for col in btree_columns:
        create_btree(dataset, col)
    for col in bitmap_columns:
        create_bitmap(dataset, col)
