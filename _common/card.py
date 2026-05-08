"""Render a Hugging Face dataset card from a few structured fields.

The output card mirrors the style of the existing lance-format cards
(``laion-1m``, ``openvid-lance``) so the org has a consistent look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CardSpec:
    repo_id: str  # e.g. "lance-format/coco-captions-2017-lance"
    pretty_name: str
    summary: str  # first line under the title
    license: str = "other"
    languages: List[str] = field(default_factory=lambda: ["en"])
    task_categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    size_category: str = "n<1K"
    splits: List[str] = field(default_factory=lambda: ["train"])
    schema_lines: List[str] = field(default_factory=list)
    indices_summary: List[str] = field(default_factory=list)
    primary_split: str = "train"
    has_image_emb: bool = False
    has_text_emb: bool = False
    has_fts: bool = False
    citation: Optional[str] = None
    extra_sections: List[str] = field(default_factory=list)


def render(spec: CardSpec) -> str:
    """Render the spec to a markdown string suitable for ``README.md``."""
    yaml_lines = ["---"]
    yaml_lines.append(f"license: {spec.license}")
    if spec.task_categories:
        yaml_lines.append("task_categories:")
        for t in spec.task_categories:
            yaml_lines.append(f"- {t}")
    if spec.languages:
        yaml_lines.append("language:")
        for lg in spec.languages:
            yaml_lines.append(f"- {lg}")
    if spec.tags:
        yaml_lines.append("tags:")
        for tag in spec.tags:
            yaml_lines.append(f"- {tag}")
    yaml_lines.append(f"pretty_name: {spec.pretty_name}")
    yaml_lines.append("size_categories:")
    yaml_lines.append(f"- {spec.size_category}")
    yaml_lines.append("---")
    yaml = "\n".join(yaml_lines)

    primary_lance_url = f"hf://datasets/{spec.repo_id}/data/{spec.primary_split}.lance"
    db_url = f"hf://datasets/{spec.repo_id}/data"

    body = []
    body.append(f"# {spec.pretty_name}\n")
    body.append(f"{spec.summary}\n")

    body.append("## Key features\n")
    feats = [
        "All multimodal data (images, text, embeddings) stored **inline** in the same Lance dataset — no sidecar files.",
        "Lance columnar layout with random-access reads — scan metadata without touching image bytes, then fetch images on demand.",
        "Hugging Face Hub integration — open with `lance.dataset(\"hf://datasets/...\")`.",
    ]
    if spec.has_image_emb or spec.has_text_emb:
        feats.append("Pre-computed embeddings with an IVF_PQ ANN index for instant similarity search.")
    if spec.has_fts:
        feats.append("Full-text inverted index on caption / question text.")
    body.append("\n".join(f"- {f}" for f in feats) + "\n")

    body.append("## Splits\n")
    body.append(", ".join(f"`{s}.lance`" for s in spec.splits) + "\n")

    body.append("## Load with `datasets.load_dataset`\n")
    body.append(
        "```python\n"
        "import datasets\n\n"
        f"hf_ds = datasets.load_dataset(\"{spec.repo_id}\", split=\"{spec.primary_split}\", streaming=True)\n"
        "for row in hf_ds.take(3):\n"
        "    print({k: v for k, v in row.items() if k != 'image'})\n"
        "```\n"
    )

    body.append("## Load directly with Lance (recommended)\n")
    body.append(
        "```python\n"
        "import lance\n\n"
        f"ds = lance.dataset(\"{primary_lance_url}\")\n"
        "print(ds.count_rows(), ds.schema.names)\n"
        "print(ds.list_indices())\n"
        "```\n"
    )

    body.append("## Load with LanceDB\n")
    body.append(
        "```python\n"
        "import lancedb\n\n"
        f"db = lancedb.connect(\"{db_url}\")\n"
        f"tbl = db.open_table(\"{spec.primary_split}\")\n"
        "print(len(tbl))\n"
        "```\n"
    )

    body.append(
        "> **Tip — for production use, download locally first.** "
        "Streaming from the Hub works for exploration, but heavy random access "
        "and ANN search are far faster against a local copy:\n"
        "> ```bash\n"
        f"> hf download {spec.repo_id} --repo-type dataset --local-dir ./{spec.repo_id.split('/')[-1]}\n"
        "> ```\n"
        f"> Then `lance.dataset(\"./{spec.repo_id.split('/')[-1]}/data/{spec.primary_split}.lance\")`.\n"
    )

    body.append("## Schema\n")
    if spec.schema_lines:
        body.append("\n".join(f"- {l}" for l in spec.schema_lines) + "\n")

    if spec.indices_summary:
        body.append("## Pre-built indices\n")
        body.append("\n".join(f"- {l}" for l in spec.indices_summary) + "\n")

    if spec.has_image_emb or spec.has_text_emb:
        body.append("## Vector search example\n")
        col = "image_emb" if spec.has_image_emb else "text_emb"
        body.append(
            "```python\n"
            "import lance\n"
            "import pyarrow as pa\n\n"
            f"ds = lance.dataset(\"{primary_lance_url}\")\n"
            f"emb_field = ds.schema.field(\"{col}\")\n"
            f"ref = ds.take([0], columns=[\"{col}\"]).to_pylist()[0][\"{col}\"]\n"
            "query = pa.array([ref], type=emb_field.type)\n\n"
            "neighbors = ds.scanner(\n"
            "    nearest={\n"
            "        \"column\": emb_field.name,\n"
            "        \"q\": query[0],\n"
            "        \"k\": 5,\n"
            "        \"nprobes\": 16,\n"
            "        \"refine_factor\": 30,\n"
            "    },\n"
            ").to_table().to_pylist()\n"
            "for n in neighbors:\n"
            "    print(n.get(\"caption\") or n.get(\"label_name\"))\n"
            "```\n"
        )

    if spec.has_fts:
        body.append("## Full-text search example\n")
        body.append(
            "```python\n"
            "import lance\n\n"
            f"ds = lance.dataset(\"{primary_lance_url}\")\n"
            "rows = ds.scanner(\n"
            "    full_text_query=\"red car on a beach\",\n"
            "    columns=[\"caption\"],\n"
            "    limit=10,\n"
            ").to_table().to_pylist()\n"
            "for r in rows:\n"
            "    print(r[\"caption\"])\n"
            "```\n"
        )

    body.append("## Working with images\n")
    body.append(
        "```python\n"
        "from pathlib import Path\n"
        "import lance\n\n"
        f"ds = lance.dataset(\"{primary_lance_url}\")\n"
        "row = ds.take([0], columns=[\"image\"]).to_pylist()[0]\n"
        "Path(\"sample.jpg\").write_bytes(row[\"image\"])\n"
        "```\n\n"
        "Images are stored inline as binary; scanning columns like `caption` or `label` does not pay the I/O cost of loading image bytes.\n"
    )

    body.append("## Why Lance?\n")
    body.append(
        "- One dataset for images + embeddings + indices + metadata — no sidecar files to manage.\n"
        "- On-disk vector and full-text indices live next to the data, so search works on local copies and on the Hub.\n"
        "- Schema evolution: add new columns (moderation labels, fresh embeddings, …) without rewriting the data ([docs](https://lance.org/guide/data_evolution/)).\n"
        "- Open standard with native Python, Rust, and SQL access.\n"
    )

    if spec.extra_sections:
        body.extend(spec.extra_sections)

    if spec.citation:
        body.append("## Citation\n")
        body.append("```\n" + spec.citation.strip() + "\n```\n")

    return yaml + "\n" + "\n".join(body)
