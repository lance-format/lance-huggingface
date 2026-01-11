import argparse
import logging
import os
from typing import List
import time
import geneva
from geneva import udf
import pyarrow as pa
import requests
from huggingface_hub import get_token

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_DIM = 384
DEFAULT_BATCH = 512
DEFAULT_CONCURRENCY = 25        
DEFAULT_MAX_FRAGMENT_GB = 18
MAX_TEXT_CHARS = 8192


def call_endpoint(url: str, payload: dict, token: str, timeout: int = 60) -> List[List[float]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list):
                raise ValueError("Invalid embedding response format")
            return result
        except Exception as exc:  # noqa: PERF203
            wait = 2 ** attempt
            logging.warning("Embedding call failed (%s). Retrying in %ss", exc, wait)
            if attempt == 2:
                raise
            time.sleep(wait)
    return []



def add_text_embeddings(
    table,
    text_column: str,
    endpoint: str,
    hf_token: str,
    dimension: int,
    batch_size: int,
    concurrency: int,
) -> None:
    if text_column not in table.schema.names:
        raise SystemExit(f"Column '{text_column}' not found in table schema")

    logging.info(
        "Backfilling text embeddings (dim=%s, batch=%s, concurrency=%s)",
        dimension,
        batch_size,
        concurrency,
    )

    @udf(data_type=pa.list_(pa.float32(), dimension), num_cpus=0.7)
    class TextEmbedding:
        def __init__(self, column: str = text_column):
            self.column = column

        def __call__(self, batch: pa.RecordBatch) -> pa.Array:
            texts = batch[self.column].to_pylist()
            normalized = [
                (text if isinstance(text, str) else "")[:MAX_TEXT_CHARS]
                for text in texts
            ]
            payload = {"inputs": normalized}
            vectors = call_endpoint(endpoint, payload, hf_token)
            if len(vectors) != len(normalized):
                raise ValueError("Embedding endpoint returned wrong batch size")
            return pa.array(vectors, pa.list_(pa.float32(), dimension))

    try:
        table.drop_columns(["text_embedding"])
        print("Dropped text_embedding column")
    except Exception:
        pass
    table.add_columns({"text_embedding": TextEmbedding()})
    table.backfill(
        "text_embedding",
        concurrency=concurrency,
        batch_size=batch_size,
    )
    logging.info("Embedding column backfilled. Current row count=%s", table.count_rows())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geneva embedding backfill for Fineweb-Edu")
    parser.add_argument("--lancedb-uri", default="./fineweb")
    parser.add_argument("--table-name", default="fineweb_edu")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--embedding-endpoint", default="https://nafn9hylwbxdvz9v.us-east-1.aws.endpoints.huggingface.cloud")
    parser.add_argument("--embedding-dimension", type=int, default=DEFAULT_DIM)
    parser.add_argument("--embedding-batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--embedding-concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-fragment-gb", type=float, default=DEFAULT_MAX_FRAGMENT_GB)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        raise SystemExit("HF token required for endpoint access")

    db = geneva.connect(args.lancedb_uri)
    table = db.open_table(args.table_name)

    add_text_embeddings(
        table,
        args.text_column,
        args.embedding_endpoint,
        hf_token,
        args.embedding_dimension,
        args.embedding_batch,
        args.embedding_concurrency,
    )

    logging.info("Finished text embedding backfill")


if __name__ == "__main__":
    main()
