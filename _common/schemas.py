"""Schema helpers for Lance datasets."""

from __future__ import annotations

import pyarrow as pa


def blob_field(name: str, *, nullable: bool = False) -> pa.Field:
    """LargeBinary field tagged as a Lance blob (legacy 2.1 metadata)."""
    return pa.field(
        name,
        pa.large_binary(),
        nullable=nullable,
        metadata={b"lance-encoding:blob": b"true"},
    )


def fixed_size_emb_field(name: str, dim: int, *, nullable: bool = False) -> pa.Field:
    """Fixed-size embedding column (float32). Use for IVF_PQ vector indices."""
    return pa.field(name, pa.list_(pa.float32(), dim), nullable=nullable)
