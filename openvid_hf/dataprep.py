#!/usr/bin/env python3
"""
OpenVid Direct Ingestion to Lance with Video Blobs.

PREREQUISITES (not covered in this script):
1. Source videos stored in S3 bucket
2. LanceDB table with Pre-computed embeddings using Geneva (or similar embedding model)

This pipeline:
- Streams metadata from source LanceDB table
- Downloads videos from S3 in parallel
- Stores videos as inline blobs in Lance format
- Creates vector (IVF_PQ) and FTS (INVERTED) indices
-  pushes to HuggingFace Hub

For consuming the dataset, see examples.py instead.
"""

import os
import argparse
import shutil
import lance
import lancedb
import pyarrow as pa
import boto3
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi


ENTERPRISE_DB_URI = os.getenv("LANCEDB_URI", "db://your-database")
ENTERPRISE_API_KEY = os.getenv("LANCEDB_API_KEY", "")
ENTERPRISE_HOST = os.getenv("LANCEDB_HOST", "")
ENTERPRISE_REGION = os.getenv("LANCEDB_REGION", "us-east-1")


def get_output_schema() -> pa.Schema:
    return pa.schema([
        pa.field("video_path", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("aesthetic_score", pa.float64()),
        pa.field("motion_score", pa.float64()),
        pa.field("temporal_consistency_score", pa.float64()),
        pa.field("camera_motion", pa.string()),
        pa.field("frame", pa.int64()),
        pa.field("fps", pa.float64()),
        pa.field("seconds", pa.float64()),
        pa.field("embedding", pa.list_(pa.float32(), 1024)),
        pa.field("video_blob", pa.large_binary(), 
                 metadata={"lance-encoding:blob": "true"}),
    ])


class S3VideoDownloader:
    def __init__(self, access_key=None, secret_key=None, region="us-east-1"):
        if access_key and secret_key:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
        else:
            self.s3 = boto3.client("s3", region_name=region)
    
    def download_bytes(self, s3_url: str) -> bytes:
        if not s3_url.startswith("s3://"):
            return b""
        
        parts = s3_url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

def record_batch_iterator(
    enterprise_table,
    s3_downloader: S3VideoDownloader,
    schema: pa.Schema,
    batch_size: int = 10,
    num_workers: int = 16,
    limit: int = None,
    offset: int = 0
) -> Iterator[pa.RecordBatch]:
    total_processed = 0
    current_offset = offset
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while True:
            rows_to_fetch = batch_size
            if limit and total_processed + batch_size > limit:
                rows_to_fetch = limit - total_processed
                
            if rows_to_fetch <= 0:
                break

            query = enterprise_table.search().limit(rows_to_fetch).offset(current_offset)
            batch_df = query.to_pandas()
            
            if len(batch_df) == 0:
                break
            
            # Parallel download of video bytes
            urls = batch_df["video"].tolist()
            video_blobs = list(executor.map(s3_downloader.download_bytes, urls))
            
            data = {k: [] for k in schema.names}
            
            for i, (_, row) in enumerate(batch_df.iterrows()):
                data["video_path"].append(row["video"])
                data["caption"].append(row["caption"])
                data["aesthetic_score"].append(row["aesthetic score"])
                data["motion_score"].append(row["motion score"])
                data["temporal_consistency_score"].append(row["temporal consistency score"])
                data["camera_motion"].append(row["camera motion"])
                data["frame"].append(row["frame"])
                data["fps"].append(row["fps"])
                data["seconds"].append(row["seconds"])
                data["embedding"].append(row["embedding"])
                data["video_blob"].append(video_blobs[i])
                
                total_processed += 1
                if total_processed % 50 == 0:
                    print(f"  Processed {total_processed} items...")

            yield pa.RecordBatch.from_pydict(data, schema=schema)
            current_offset += len(batch_df)
def main():
    parser = argparse.ArgumentParser(description="OpenVid Dataprep: Enterprise -> Lance Blobs")
    parser.add_argument("-o", "--output", type=str, default="./openvid.lance", help="Output path")
    parser.add_argument("--limit", type=int, help="Total items to process")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for ingestion")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of parallel download workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset")
    parser.add_argument("--index", action="store_true", default=False, help="Create Vector and FTS indices")
    parser.add_argument("--no-index", action="store_false", dest="index", help="Skip indexing")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HF Hub after finished")
    parser.add_argument("--repo-id", type=str, default="lance-format/openvid-lance", help="HF Repo ID")
    parser.add_argument("--token", type=str, help="HF Token")
    args = parser.parse_args()

    aws_access = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access or not aws_secret:
        print("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return
    
    print(f"Starting ingestion to {args.output} (Workers: {args.num_workers})")
    
    db = lancedb.connect(
        uri=ENTERPRISE_DB_URI,
        api_key=ENTERPRISE_API_KEY,
        host_override=ENTERPRISE_HOST,
        region=ENTERPRISE_REGION,
    )
    table = db.open_table("openvid")
    
    s3 = S3VideoDownloader(aws_access, aws_secret)
    schema = get_output_schema()
    
    if args.overwrite and os.path.exists(args.output):
        shutil.rmtree(args.output)
        
    # 15GB in bytes
    MAX_BYTES_PER_FILE = 15 * 1024 * 1024 * 1024
    
    if os.path.exists(args.output):
        print(f"Dataset already exists at {args.output}. Skipping ingestion.")
        dataset = lance.dataset(args.output)
    else:
        print(f"Creating new dataset at {args.output}")
        dataset = lance.write_dataset(
            record_batch_iterator(
                table, s3, schema, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                limit=args.limit, 
                offset=args.offset
            ),
            args.output,
            schema=schema,
            mode="create",
            max_bytes_per_file=MAX_BYTES_PER_FILE
        )
        print(f"Ingestion complete.")

    if args.index:
        print(" Creating Vector Index ")
        dataset.create_index(
            column="embedding",
            index_type="IVF_PQ",
            num_partitions=256,
            num_sub_vectors=64,
            replace=True
        )
        
        print(" Creating FTS Index on caption.")
        tbl = lancedb.connect(os.path.dirname(args.output))[os.path.basename(args.output).replace(".lance", "")]
        tbl.create_fts_index("caption", replace=True)
        
        print("Indexing complete.")

    if args.push_to_hub:
        token = args.token or os.getenv("HF_TOKEN")
        
        if not token:
            print("HF_TOKEN not found. Set --token or HF_TOKEN environment variable")
            return
        
        repo_id = args.repo_id
        
        print(f"Pushing to HF Hub: {repo_id}")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=args.output,
            repo_type="dataset",
        )
        print(f"Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()