import argparse
import time
import os
import pyarrow as pa
import lancedb
from datasets import load_dataset
from typing import Optional, Iterator
from datetime import datetime
from threading import Thread, Event

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "default"

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} - INFO - {msg}", flush=True)


def make_reader(data_iterator: Iterator[pa.Table], limit: Optional[int]) -> Iterator[pa.RecordBatch]:
    """
    Converts the HuggingFace Table iterator into a RecordBatch iterator
    which LanceDB consumes with zero-copy.
    """
    total_yielded = 0
    for table in data_iterator:
        for batch in table.to_batches():
            if limit and total_yielded >= limit:
                return
            
            # Trim the last batch if it exceeds the limit
            if limit and (total_yielded + len(batch)) > limit:
                batch = batch.slice(0, limit - total_yielded)
                yield batch
                return
            
            yield batch
            total_yielded += len(batch)

def main():
    parser = argparse.ArgumentParser(description="High-Performance Iterator Ingester")
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    parser.add_argument("--lancedb-uri", default="./fineweb")
    parser.add_argument("--table-name", default="fineweb_edu")
    parser.add_argument("--batch-size", type=int, default=100_000) # Size of chunks fetched from HF
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    db = lancedb.connect(args.lancedb_uri)
    
    if args.overwrite and args.table_name in db.table_names():
        log(f"Dropping existing table '{args.table_name}'...")
        db.drop_table(args.table_name)

    log(f"Initializing HF Stream...")
    ds = load_dataset(DATASET_NAME, args.dataset_config, split="train", streaming=True).with_format("arrow")
    
    hf_iterator = ds.iter(batch_size=args.batch_size)
    record_batch_reader = make_reader(hf_iterator, args.limit)

    start_time = time.time()    
    db.create_table(
        args.table_name, 
        data=record_batch_reader, 
        mode="overwrite" if args.overwrite else "create"
    )
    total_time = time.time() - start_time
    final_tbl = db.open_table(args.table_name)
    final_count = final_tbl.count_rows()
    
    log(f"Total Rows: {final_count:,}")
    log(f"Total Time: {total_time:.2f}s")
    log(f"Overall Speed: {final_count / total_time:.2f} rows/s")

if __name__ == "__main__":
    main()