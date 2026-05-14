# Notes for Agents

## Dataset documentation audience

- Treat `README.md` files as maintainer-facing unless explicitly stated otherwise.
- Treat `HF_DATASET_CARD.md` as public user-facing documentation for Hugging Face dataset consumers.
- Do not copy maintainer workflows (data generation, conversion internals, upload mechanics) into `HF_DATASET_CARD.md` unless explicitly requested.

## Writing HF dataset cards

When creating or revising an `HF_DATASET_CARD.md`, follow the conventions in [skills/hf-dataset-card-writer/SKILL.md](skills/hf-dataset-card-writer/SKILL.md). That skill defines the six-section body structure (Search, Curate, Evolve, Train, Versioning, Materialize a subset), the writing voice, the example style rules (path handling, materialization discipline, LanceDB-first idioms, SQL-first column derivation), and links to the current canonical reference card.
