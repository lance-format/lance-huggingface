#!/usr/bin/env python3
#

import argparse
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.json as pa_json
import lance
from tqdm import tqdm


def cohere_fineweb(args):
    def data_gen():
        all_np_files = list(Path("./data/fineweb-edu-emb/emb").glob("*.npy"))
        for npfile in tqdm(all_np_files):
            stem = npfile.stem
            corpus_path = Path(
                "./data/fineweb-edu-corpus/corpus", f"{stem}.parquet.jsonl.zst"
            )
            tbl = pa_json.read_json(corpus_path)
            print(tbl.num_rows)
            embeddings = np.load(npfile).astype(np.float32)
            assert tbl.num_rows == embeddings.shape[0]
            dim = embeddings.shape[1]

            emb_arr = pa.FixedSizeListArray.from_arrays(embeddings.reshape(-1), dim)
            tbl = tbl.append_column("cohere_emb", emb_arr)
            yield from tbl.to_batches()

    schema = next(data_gen()).schema
    shutil.rmtree("fineweb-edu-cohere.lance", ignore_errors=True)

    lance.write_dataset(
        data_gen(),
        "fineweb-edu-cohere/data/train.lance",
        schema=schema,
        data_storage_version="2.1",
        max_bytes_per_file=15_000_000_000,
    )
    shutil.copy("./README.cohere_emb.md", "fineweb-edu-cohere/README.md")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    cohere_fineweb(args)


if __name__ == "__main__":
    main()
