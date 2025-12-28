# Disk Usage Estimate: Fineweb-Edu (1.5B Rows)

## Summary
**Estimated Total Space Required:** ~6 TB - 8 TB
*(Significantly reduced from previous estimates due to streaming and no chunking)*

## Breakdown

### 1. Raw Text Data (Streaming)
- **Source:** Streaming directly from HF (No local Parquet storage required).
- **Scale:** ~1.3 Trillion tokens across 1.5 Billion rows.
- **Estimated Text Size:** ~3.0 TB - 4.0 TB
  - Based on ~2-3 bytes per token compressed in Lance.
  - 1.3T tokens * 2.5 bytes ≈ 3.25 TB.

### 2. Embeddings (1 vector per row)
- **Count:** 1.5 Billion vectors (No chunking).
- **Dims:** 384 dimensions (float32).
- **Size per Vector:** 384 * 4 bytes = 1,536 bytes (~1.5 KB).
- **Total Vector Data:** 1.5B * 1.536 KB ≈ **2.3 TB**.

### 3. Indices & Overhead
- **Vector Index (IVF_PQ):** ~10-15% of vector data ≈ **0.3 TB**.
- **Metadata/Overhead:** ~0.2 TB.

### Total Calculation
**3.5 TB (Text) + 2.3 TB (Vectors) + 0.5 TB (Index/Overhead) ≈ 6.3 TB**

> **Recommendation:** Keep at least **8 TB** free to handle temporary files during indexing and compaction.
