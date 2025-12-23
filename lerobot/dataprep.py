#!/usr/bin/env python3

import argparse
import os
import shutil

import datasets
import lance
from pathlib import Path


def size_catagory(num_rows: int) -> str:
    # TODO: should move this to a common util file
    match num_rows:
        case n if n < 1_000:
            return "n<1K"
        case n if n < 10_000:
            return "1K<n<10K"
        case n if n < 100_000:
            return "10K<n<100K"
        case n if n < 1_000_000:
            return "100K<n<1M"
        case n if n < 10_000_000:
            return "1M<n<10M"
        case n if n < 100_000_000:
            return "10M<n<100M"
        case n if n < 1_000_000_000:
            return "100M<n<1B"
        case n if n < 10_000_000_000:
            return "1B<n<10B"
        case n if n < 100_000_000_000:
            return "10B<n<100B"
        case n if n < 1_000_000_000_000:
            return "100B<n<1T"
        case _:
            return "n>1T"


def generate_hf_dataset(
    hf_dataset_dict: datasets.DatasetDict, dataset_name: str, output_base_path: Path
):
    for split in hf_dataset_dict:
        hf_ds = hf_dataset_dict[split]
        lance.write_dataset(
            hf_ds.rename_column("observation.state", "observation_state"),
            output_base_path / "data" / f"{split}.lance",
        )

    total_rows =  sum(len(hf_dataset_dict[split]) for split in hf_dataset_dict)
    with open(output_base_path / "README.md", "w") as f:
        f.write(
            f"""---
tags:
- lerobot
- lance
pretty_name: LeRobot Dataset
size_categories:
 - {size_catagory(total_rows)}
source_datasets:
 - {dataset_name}
---
# LeRobot dataset converted from Hugging Face\n\n
Original dataset: `{dataset_name}`\n"
"""
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset from Hugging Face to Lance format"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output Lance file path", metavar="PATH"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="hf",
        choices=["hf", "lance"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="lerobot/xvla-soft-fold",
        help="Huggingface dataset repo (default: lerobot/xvla-soft-fold)",
        metavar="ORG/REPO",
    )
    args = parser.parse_args()

    dataset_name = os.path.split(args.dataset)[1]
    hf_dataset_dict = datasets.load_dataset(args.dataset)
    output_base_path = (
        Path(args.output) if args.output else Path("lerobot_" + dataset_name)
    )
    shutil.rmtree(output_base_path, ignore_errors=True)

    match args.format:
        case "hf":
            generate_hf_dataset(hf_dataset_dict, args.dataset, output_base_path)
        case _:
            raise ValueError(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    main()
