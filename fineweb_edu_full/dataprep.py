import argparse
import logging
import time
import os
import requests
import pyarrow as pa
import geneva
from geneva import udf
from datasets import load_dataset
from typing import Iterator, Dict, Optional, Any
from huggingface_hub import HfApi, get_token
from datetime import datetime

# "default" corresponds to the full dataset (~1.5T tokens / ~1.5B rows)
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "default"

logging.basicConfig(level=logging.CRITICAL)

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - INFO - {msg}")

def log_error(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - ERROR - {msg}")


def ingest_stream(
    table,
    iterator: Iterator,
    limit: Optional[int] = None,
) -> int:
    total_rows = 0
    start_time = time.time()
    last_ts = start_time
    
    for arrow_table in iterator:
        if limit is not None and total_rows >= limit:
            break
        
        if limit is not None and (total_rows + len(arrow_table)) > limit:
            arrow_table = arrow_table.slice(0, limit - total_rows)

        table.add(arrow_table)
        current_batch_size = len(arrow_table)
        total_rows += current_batch_size
        
        now = time.time()
        interval_duration = now - last_ts
        
        if interval_duration > 0:
            current_speed = current_batch_size / interval_duration
            avg_speed = total_rows / (now - start_time)
            log(f"Inserted {total_rows:,} rows | Current: {current_speed:.2f} rows/s | Avg: {avg_speed:.2f} rows/s")
        
        last_ts = now

    return total_rows

def call_endpoint(url: str, payload: Dict, token: str) -> Any:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 503:
                time.sleep(10)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 4: raise e
            time.sleep(2 * (attempt + 1))

def add_embeddings(tbl, column_name, endpoint, hf_token, batch_size=1024, concurrency=10):
    log(f"Backfilling embedding column '{column_name}'...")
    start_time = time.time()
    
    @udf(data_type=pa.list_(pa.float32(), 384))
    class TextEmbedding:
        def __init__(self, endpoint_url=endpoint, token=hf_token):
            self.endpoint = endpoint_url
            self.token = token
        def __call__(self, batch: pa.RecordBatch):
            content = batch["text"].to_pylist()
            payload = {"inputs": content}
            embeddings = call_endpoint(self.endpoint, payload, self.token)
            return pa.array(embeddings, pa.list_(pa.float32(), 384))

    try:
        tbl.drop_columns([column_name])
    except:
        pass
        
    tbl.add_columns({column_name: TextEmbedding()})
    tbl.backfill(column_name, batch_size=batch_size, concurrency=concurrency, commit_granularity=1000)
    
    log(f"Encoding finished in {time.time() - start_time:.2f}s")
    log("Creating vector index...")
    tbl.create_index(vector_column_name=column_name, index_type="IVF_PQ", num_partitions=256, num_sub_vectors=96)
    log("Creating FTS index...")
    tbl.create_fts_index("text", replace=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    parser.add_argument("--lancedb-uri", default="./fineweb")
    parser.add_argument("--table-name", default="fineweb_edu")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embedding-endpoint", default="https://placeholder.url")
    parser.add_argument("--embedding-column", default="embedding")
    parser.add_argument("--embedding-batch", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default="lance-format/fineweb_edu")
    parser.add_argument("--token")
    args = parser.parse_args()
    
    hf_token = args.token or get_token() or os.getenv("HF_TOKEN")
    db = geneva.connect(args.lancedb_uri)
    
    run_ingestion = not args.embed_only
    run_embedding = not args.ingest_only

    tbl = None
    try:
        tbl = db.open_table(args.table_name)
        log(f"Opened table with {tbl.count_rows()} rows.")
        if args.overwrite and run_ingestion:
            db.drop_table(args.table_name)
            raise FileNotFoundError
    except:
        if run_ingestion: log(f"Creating new table '{args.table_name}'")
    
    if run_ingestion and (not tbl or args.overwrite):
        log(f"Streaming dataset {DATASET_NAME}...")
        
        ds_stream = load_dataset(
            DATASET_NAME, args.dataset_config, split="train", streaming=True
        ).with_format("arrow")
            
        iterator = ds_stream.iter(batch_size=args.batch_size)
        
        log("Fetching initial batch for initialization...")
        try:
            first_batch = next(iterator)
        except StopIteration:
            log_error("Dataset is empty.")
            return

        tbl = db.create_table(args.table_name, data=first_batch, mode="overwrite")
        
        remaining_limit = (args.limit - len(first_batch)) if args.limit else None
        if remaining_limit is None or remaining_limit > 0:
            ingest_stream(tbl, iterator, limit=remaining_limit)

        log(f"Ingestion done. Total rows: {tbl.count_rows()}")

    if run_embedding:
        if not tbl:
            tbl = db.open_table(args.table_name)
        add_embeddings(tbl, args.embedding_column, args.embedding_endpoint, hf_token, args.embedding_batch, args.concurrency)

    if args.push_to_hub:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(repo_id=args.repo_id, folder_path=args.lancedb_uri, repo_type="dataset")
        log("Upload complete!")

if __name__ == "__main__":
    main()