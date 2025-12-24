#!/usr/bin/env python3

from huggingface_hub import HfApi
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Hugging Face dataset path")
    args = parser.parse_args()

    hf = HfApi()

    repo_id = "lance-format/" + os.path.basename(args.dataset_path)
    print(f"Creating repository: {repo_id}")

    hf.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    hf.upload_folder(
        repo_id=repo_id,
        folder_path=args.dataset_path,
        repo_type="dataset",
    )
