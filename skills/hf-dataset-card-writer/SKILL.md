---
name: hf-dataset-card-writer
description: Use when creating or revising an HF_DATASET_CARD.md file in this repo. Defines the six-section card structure (Search, Curate, Evolve, Train, Versioning, Materialize a subset), the writing voice, and the example style rules for Lance-format datasets published to the HuggingFace Hub.
---

# HF dataset card writer

This skill governs the public-facing `HF_DATASET_CARD.md` published with each Lance-format dataset on the HuggingFace Hub. Treat the card as user-facing documentation for ML engineers, researchers, and agent/app builders consuming the dataset. Do not copy maintainer workflows (data generation, conversion internals, upload mechanics) into the card unless explicitly requested.

## Audience and framing

Examples should make the value of Lance self-evident — show, don't tell. Never name or compare to other formats (Parquet, WebDataset, image folders); demonstrate the strengths and let the reader draw their own conclusion. Do not label sections with persona tags ("for agents", "for researchers", etc.) — the same section often applies to multiple personas, and readers should not have to self-identify before reading.

## Writing style

Prose should be informative and educational, written in full, flowing sentences that explain what the code does and why it matters. Avoid the staccato style of fragments stitched together with em-dashes, parenthetical asides, and bullet-like one-liners; readers should be able to learn the concept from the prose alone, with the code as confirmation. Each section's introduction should set up the problem the snippet addresses, name the relevant Lance mechanism, and explain the consequence of using it. Short follow-up paragraphs after a snippet should expand on a non-obvious property of the result rather than restate what the code did.

Bulleted lists, terse callouts, and tables remain appropriate for the inherently list-shaped parts of the card (key features, schema, indices, tips). The prose-style rule applies to the body of each of the six sections below. Write in an informative, educational manner without being verbose; use em-dashes sparingly.

## Card structure

The top of every card contains, in order:

1. YAML frontmatter (license, task_categories, language, tags, pretty_name, size_categories)
2. Title and one-paragraph summary
3. Key features (bullet list)
4. Splits
5. Schema (table with column names, types, notes — not a raw PyArrow dump)
6. Pre-built indices
7. Loaders — in order: `datasets.load_dataset`, `lancedb.connect`, then `lance.dataset`. LanceDB comes before pylance because it is the higher-level interface most users interact with; pylance is shown below as the format-level handle for readers who want to inspect or operate on dataset internals.
8. The "for production use, download locally first" tip with the `hf download` command

Keep `meta/info.json` in the dataset card because users rely on it for global metadata and feature definitions. Explain the purpose of each table in plain language (what it is for, when to use it), not only terse schema bullets.

Every card then carries six body sections in this order:

### 1. Search

Show a regular vector search using the dataset's bundled vector index. If the dataset also ships an FTS index, follow up with a hybrid search (vector + FTS) example. Keep this section focused on retrieval mechanics; versioning lives in section 5. All six sections appear in every card, even when one feels forced for the modality.

### 2. Curate

Ad-hoc filtering and slicing that produces a small, explicit candidate set for the next experiment. Always bounded by `.limit(...)`. Output is framed as a row list that feeds into Evolve or Train.

### 3. Evolve

Add features without rewriting the dataset. Prefer `tbl.add_columns({"new_col": "SQL_EXPRESSION"})` for in-card examples: a SQL expression that derives the new column from existing ones is a one-liner with no decorators, no Arrow construction, and no fake placeholder values. Pair it with `tbl.merge(other, on="key")` to attach offline labels or predictions from an external table, if required. Mention that original columns and indices are untouched.

### 4. Train

Feed a PyTorch loop with column projection. Use `lance.torch.data.LanceDataset` wrapped in `torch.utils.data.DataLoader`. The `columns=[...]` argument is the lever: only the projected columns are read per epoch, so columns added in Evolve cost nothing until you opt in.

### 5. Versioning

Show how to list versions / tags and open a pinned snapshot (`db.open_table(..., version=...)` or via tags). Cover both use cases without labeling personas: looking up past data versions for added context at serving time, and running experiments with different models against different data versions. This is where the reproducibility and governance story lives.

### 6. Materialize a subset

Stream a filtered query from the Hub-mounted table into a new local LanceDB table using `tbl.search().where(...).select([...]).to_batches()` piped into `db.create_table(name, batches)`. Explain that reads from the Hub are already lazy, that mutations and fast training need a writable local store, and that this pattern transfers only the projected columns and matching row groups without ever fully materializing in Python memory. State explicitly that the Evolve, Train, and Versioning snippets above can target the local subset by swapping the path. This is the least essential of the six sections — it sits at the end as a reference pattern that earlier sections can point back to when readers need a writable copy.

## Example style rules

- **Never materialize full result sets in memory.** Do not call `to_table()`, `to_pylist()` (without a bounded `take` / `limit`), `to_arrow()`, or `to_pandas()` on a scanner or search in examples — readers will copy these against million-row datasets and OOM or stall. Bounded `ds.take([id1, id2, ...])` is fine because it's index-driven.
- **Use LanceDB APIs for Search and Curate.** `db.open_table(...)` then `tbl.search(...).where(...).select([...]).limit(k).to_list()`. `.limit(k)` is mandatory before any **in-memory** terminal (`.to_list()`, `.to_arrow()`, `.to_pandas()`).
- **Streaming terminals are exempt from the `.limit()` rule.** `.to_batches()` returns a `pa.RecordBatchReader` and does not accumulate rows in Python memory, so it is safe (and expected) without a `.limit()` when the goal is to pipe a filtered query into another sink — typically `db.create_table(name, batches)` in the Materialize-a-subset section.
- **Keep exactly one pylance entry-point snippet at the top** of the card (the `lance.dataset(...)` + `count_rows()` + `list_indices()` block) so readers see the low-level handle. Use pylance for **Train** (where `lance.torch.data.LanceDataset` lives). Use LanceDB everywhere else, including Evolve and Versioning.
- **Use the dataset's real column names** in every example, not generic placeholders. Examples should be runnable copy-paste against the published dataset.
- **Point dataset paths at the Hub by default.** Card examples should use `hf://datasets/<org>/<repo>/data` (LanceDB) or `hf://datasets/<org>/<repo>/data/<split>.lance` (pylance) so a reader can run the code without downloading anything first. Use a local path (e.g., `./<repo>/data`) only in snippets that mutate the dataset — `add_columns`, `merge`, `create_index`, `tags.create` — because the Hub mount is read-only. When introducing a local path, add a one-line note explaining why and pointing back to the `hf download` tip (or the Materialize-a-subset section) near the top or bottom of the card.
- **Prefer SQL expressions over Python.** SQL filter strings in `.where(...)` and SQL derivation expressions in `add_columns({col: sql_expr})` are the universal idiom and require no extra imports or decorators. Lean on SQL anywhere the operation can be expressed declaratively.

## Reference

The current canonical example of a card written to this spec is [`laion-1M/HF_DATASET_CARD.md`](../../laion-1M/HF_DATASET_CARD.md). Read it before writing or revising another card.
