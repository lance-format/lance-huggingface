import argparse
import logging
import time
import os
import shutil
import requests
import pyarrow as pa
import geneva
from geneva import udf
from datasets import load_dataset
from typing import Iterable, Iterator, List, Dict, Optional, Any
from huggingface_hub import HfApi, get_token
from datetime import datetime
import multiprocessing
import torch
from torch.utils.data import DataLoader

# "default" corresponds to the full dataset (~1.5T tokens / ~1.5B rows)
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "default"


logging.basicConfig(level=logging.CRITICAL)

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - {msg}")

def log_error(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - ERROR - {msg}")

def batched(iterator: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def ingest_stream(
    table,
    iterator: Iterator,
    batch_size: int = 100_000,
    limit: Optional[int] = None,
) -> int:
    total_rows = 0
    start_time = time.time()
    
    for batch in batched(iterator, batch_size):
        if limit is not None and total_rows >= limit:
            break
        
        table.add(batch)
        total_rows += len(batch)
        
        if total_rows % batch_size == 0:
            elapsed = time.time() - start_time
            rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
            log(f"Inserted {total_rows} rows... ({rows_per_sec:.2f} rows/s)")

    return total_rows

def call_endpoint(url: str, payload: Dict, token: str) -> Any:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 503: # Model loading
                time.sleep(10)
                raise Exception("Model loading")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(2 * (attempt + 1))

def add_embeddings(
    tbl,
    column_name: str,
    endpoint: str,
    hf_token: str,
    batch_size: int = 1024,
    concurrency: int = 10
):
    log(f"Backfilling embedding column '{column_name}' using endpoint...")
    start_time = time.time()
    
    # Using 384 dims for BGE-small 1.5
    @udf(data_type=pa.list_(pa.float32(), 384))
    class TextEmbedding:
        def __init__(self, endpoint_url: str = endpoint, token: str = hf_token):
            self.endpoint = endpoint_url
            self.token = token

        def __call__(self, batch: pa.RecordBatch):
            # Endpoint handles truncation (max_length=512) via tokenizer
            content = batch["text"].to_pylist()
            payload = {"inputs": content}
            
            try:
                embeddings = call_endpoint(self.endpoint, payload, self.token)
                return pa.array(embeddings, pa.list_(pa.float32(), 384))
            except Exception as e:
                log_error(f"Text embedding failed: {e}")
                raise e

    try:
        tbl.drop_columns([column_name])
    except Exception:
        pass
        
    tbl.add_columns({column_name: TextEmbedding()})
    
    tbl.backfill(
        column_name,
        batch_size=batch_size,
        concurrency=concurrency,
        commit_granularity=1000,
    )
    
    elapsed = time.time() - start_time
    log(f"Encoding finished in {elapsed:.2f}s")
    
    log(f"Creating vector index for {column_name}")
    try:
        tbl.create_index(
            vector_column_name=column_name,
            index_type="IVF_PQ", 
            num_partitions=256, 
            num_sub_vectors=96
        )
        log("Vector index created.")
    except Exception as e:
        log_error(f"Failed to create vector index: {e}")

    log("Creating FTS index on 'text' column...")
    try:
        tbl.create_fts_index("text", replace=True)
        log("FTS index created.")
    except Exception as e:
        log_error(f"Failed to create FTS index: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ingest Fineweb-Edu to LanceDB with HF Inference Endpoint Embeddings")
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG, help="HF Dataset config")
    parser.add_argument("--lancedb-uri", default="./fineweb", help="Local LanceDB URI")
    parser.add_argument("--table-name", default="fineweb_edu", help="Table name")
    parser.add_argument("--batch-size", type=int, default=20000, help="Ingestion batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to ingest")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing table")
    
    parser.add_argument("--embedding-endpoint", default="https://placeholder-endpoint-url.us-east-1.aws.endpoints.huggingface.cloud", help="HF Inference Endpoint URL")
    parser.add_argument("--embedding-column", default="embedding")
    parser.add_argument("--embedding-batch", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--ingest-only", action="store_true", help="Run only ingestion")
    parser.add_argument("--embed-only", action="store_true", help="Run only embedding backfill")
    
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default="lance-format/fineweb_edu", help="Target HF Repo ID")
    parser.add_argument("--token", help="HF Token")
    
    parser.add_argument("--num-workers", type=int, default=16, help="Number of dataloader workers (8-16 recommended for 64GB RAM)")
    parser.add_argument("--prefetch-factor", type=int, default=8, help="Internal prefetch factor for dataloader (2-10 recommended)")
    
    args = parser.parse_args()
    
    hf_token = args.token or get_token() or os.getenv("HF_TOKEN")
    if not hf_token and (args.push_to_hub or not args.ingest_only):
        log("WARNING: No HF token found. Some operations may fail.")

    run_ingestion = not args.embed_only
    run_embedding = not args.ingest_only

    db = geneva.connect(args.lancedb_uri)
    
    tbl = None
    try:
        tbl = db.open_table(args.table_name)
        log(f"Opened existing table '{args.table_name}' with {tbl.count_rows()} rows.")
        if args.overwrite and run_ingestion:
            log("Overwrite set. Dropping table.")
            db.drop_table(args.table_name)
            raise FileNotFoundError
    except Exception:
        if run_ingestion:
            log(f"Creating new table '{args.table_name}'")
    
    if run_ingestion and (not tbl or args.overwrite):
        log(f"Streaming dataset {DATASET_NAME}/{args.dataset_config}...")
        ds_stream = load_dataset(DATASET_NAME, args.dataset_config, split="train", streaming=True)
            
        dataloader = DataLoader(
            ds_stream,
            num_workers=args.num_workers,
            batch_size=None,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
        )
        iterator = iter(dataloader)
        
        first_batch = []
        log("Fetching initial batch for table initialization...")
        for i, item in enumerate(iterator):
            first_batch.append(item)
            if i >= args.batch_size - 1:
                break
        
        if not first_batch:
            log_error("Dataset stream is empty.")
            return

        if not args.overwrite:
            tbl = db.open_table(args.table_name)
            tbl.add(first_batch)
        else:
            tbl = db.create_table(args.table_name, data=first_batch, mode="overwrite")
        
        remaining_limit = (args.limit - len(first_batch)) if args.limit else None
        if remaining_limit is None or remaining_limit > 0:
            ingest_stream(
                tbl, 
                iterator, 
                args.batch_size, 
                limit=remaining_limit, 
            )

        log(f"Ingestion done. Total rows: {tbl.count_rows()}")

    if run_embedding:
        if not tbl:
            tbl = db.open_table(args.table_name)
            
        log(f"Starting embedding with endpoint: {args.embedding_endpoint}")
        add_embeddings(
            tbl,
            column_name=args.embedding_column,
            endpoint=args.embedding_endpoint,
            hf_token=hf_token,
            batch_size=args.embedding_batch,
            concurrency=args.concurrency
        )

    if args.push_to_hub:
        if not args.repo_id:
            log_error("Repo ID required for push-to-hub")
            return
            
        log(f"Uploading to {args.repo_id}...")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=args.lancedb_uri,
            repo_type="dataset",
        )
        log("Upload complete!")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
