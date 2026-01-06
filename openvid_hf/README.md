# OpenVid HF - Lance Format with Video Blobs

This project migrates the OpenVid dataset from LanceDB Enterprise to Hugging Face Hub, storing videos inline using Lance's Blob API.

## Setup

```bash
cd openvid_hf
uv sync
```

## Usage

### Test with small batch (5-10 rows)
```bash
uv run python test_direct_ingestion.py
```

### Full migration
```bash
uv run python dataprep.py --limit 1000000
```

### Upload to HF Hub
```bash
uv run python upload_hub.py ./openvid.lance
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| video_path | string | Original S3 path |
| caption | string | Video caption |
| seconds | float64 | Duration |
| embedding | list[float32, 1024] | Video embedding |
| video | large_binary (blob) | Video bytes |
