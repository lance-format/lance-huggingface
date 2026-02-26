# Notes for Agents

## Dataset documentation audience

- Treat `README.md` files as maintainer-facing unless explicitly stated otherwise.
- Treat `HF_DATASET_CARD.md` as public user-facing documentation for Hugging Face dataset consumers.
- Do not copy maintainer workflows (data generation, conversion internals, upload mechanics) into `HF_DATASET_CARD.md` unless explicitly requested.

## HF dataset card writing rules

- Keep `meta/info.json` in the dataset card because users rely on it for global metadata and feature definitions.
- Explain the purpose of each table in plain language (what it is for, when to use it), not only terse schema bullets.
- Provide clear, readable schema summaries for each table; avoid raw PyArrow schema dumps.
- Prefer usage examples that help users consume the published dataset (download/read/sample), not maintainer pipeline steps.
