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

## Usage


### 1. Full Pipeline (Ingest + Embed + Upload)
```bash
uv run dataprep.py \
  --dataset-config "default" \
  --limit 1000 \
  --embedding-endpoint "https://your-endpoint-url.us-east-1.aws.endpoints.huggingface.cloud" \
  --push-to-hub \
  --repo-id "your-username/fineweb-edu-lance"
```

### 2. Ingestion Only
Download and convert to LanceDB format.
```bash
uv run dataprep.py \
  --ingest-only \
  --lancedb-uri "./fineweb" \
  --batch-size 100000
```

### 3. Embedding Backfill Only
Generate embeddings for an existing LanceDB table.
```bash
uv run dataprep.py \
  --embed-only \
  --lancedb-uri "./fineweb" \
  --embedding-endpoint "https://your-endpoint.example.com" \
  --concurrency 20 \
  --embedding-batch 1024
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-config` | `default` | HF Dataset config (e.g., `sample-10BT` or `default` for full). |
| `--lancedb-uri` | `./fineweb` | Local path for the LanceDB dataset. |
| `--table-name` | `fineweb_edu` | Name of the table in LanceDB. |
| `--batch-size` | `100,000` | Number of rows per batch during ingestion. |
| `--embedding-endpoint` | *Required for embedding* | URL of the HF Inference Endpoint. |
| `--embedding-batch` | `1024` | Batch size for embedding requests. |
| `--concurrency` | `10` | Number of concurrent embedding requests. |
| `--push-to-hub` | `False` | Upload the final dataset to HF Hub. |
| `--repo-id` | `lance-format/fineweb_edu` | Target Repo ID for upload. |
| `--limit` | `None` | Limit total rows (useful for testing). |
