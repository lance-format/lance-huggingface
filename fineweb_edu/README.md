# Fineweb-Edu Ingestion with Geneva

This folder contains the pipeline for ingesting the [Fineweb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset into LanceDB, generating embeddings using Geneva with Hugging Face Inference Endpoints, and uploading the result back to the Hub.

## Setup

1. **Install Dependencies**
   Ensure you have [uv](https://github.com/astral-sh/uv) installed.
   ```bash
   uv sync
   # OR
   uv pip install -r pyproject.toml
   ```

2. **Environment Variables**
   Set your Hugging Face Token (required for accessing gated datasets and endpoints):
   ```bash
   export HF_TOKEN="hf_..."
   ```
  or just make sure you are logged in to huggingface cli


1. **Ingest FineWeb-Edu into Lance** using `dataprep.py` (Arrow streaming → Lance table).
2. **Generate text embeddings** with the LanceDB enterprise feature engineering tool (Geneva) via `text_embedding_pipeline.py`.
3. **Upload the Lance folder** to the Hugging Face Hub with `hf upload-large-folder`.

Each step is explicit below so you can reproduce the uploaded `lance-format/fineweb-edu` dataset.

### 1. Ingest with `dataprep.py`

This script streams FineWeb-Edu from HF, converts to Arrow batches, and writes a Lance table.

```bash
uv run dataprep.py \
  --dataset-config default \
  --lancedb-uri ./fineweb \
  --table-name fineweb_edu \
  --batch-size 100000 \
  --limit 1000000          # optional for smoke tests
```

### 2. Backfill embeddings with Geneva (`text_embedding_pipeline.py`)

Once the Lance table exists, run Geneva to add the `text_embedding` column. This wraps the LanceDB enterprise feature engineering system and hits a Hugging Face Inference Endpoint.

```bash
uv run text_embedding_pipeline.py \
  --lancedb-uri ./fineweb \
  --table-name fineweb_edu \
  --text-column text \
  --embedding-endpoint https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud \
  --embedding-dimension 384 \
  --embedding-batch 512 \
  --embedding-concurrency 25
```

### 3. Upload Lance artifacts to Hugging Face

After embeddings land, push the `./fineweb` directory (or subfolders like `fineweb/fineweb_edu`) to the Hub. The repo used in this branch is `lance-format/fineweb-edu`.

```bash
huggingface-cli login   # or ensure HF_TOKEN is set

hf upload-large-folder \
  ./fineweb \
  lance-format/fineweb-edu \
  data
```


## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-config` | `default` | HF Dataset config (e.g., `sample-10BT` or `default` for full). |
| `--lancedb-uri` | `./fineweb` | Local path for the LanceDB dataset. |
| `--table-name` | `fineweb_edu` | Name of the table in LanceDB. |
| `--batch-size` | `100,000` | Number of rows per batch during ingestion. |
| `--embedding-endpoint` | *Required for embedding* | URL of the HF Inference Endpoint (Geneva step). |
| `--embedding-batch` | `1024` | Batch size for embedding requests. |
| `--concurrency` | `10` | Number of concurrent embedding requests. |
| `--push-to-hub` | `False` | Upload the final dataset to HF Hub. |
| `--repo-id` | `lance-format/fineweb_edu` | Target Repo ID for upload. |
| `--limit` | `None` | Limit total rows (useful for testing). |
