#!/usr/bin/env python3
"""
Debug utility: print Lance table schemas for episodes, frames, and videos.

Usage:
    uv run debug.py

    # Optional explicit paths
    uv run debug.py \
        --episodes /path/to/<repo>.episodes.lance \
        --frames /path/to/<repo>.frames.lance \
        --videos /path/to/<repo>.videos.lance
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _open_lance_dataset(path: Path):
    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance'. Install with `pip install lance`.") from e
    return lance.dataset(str(path))

def main() -> None:
    parser = argparse.ArgumentParser(description="Print Lance schemas for episodes, frames, and videos")
    parser.add_argument("--episodes", type=str, default="./lance/episodes.lance", help="Episodes Lance directory")
    parser.add_argument("--frames", type=str, default="./lance/frames.lance", help="Frames Lance directory")
    parser.add_argument("--videos", type=str, default="./lance/videos.lance", help="Videos Lance directory")
    args = parser.parse_args()

    episodes_path = Path(args.episodes).resolve()
    frames_path = Path(args.frames).resolve()
    videos_path = Path(args.videos).resolve()

    for label, path in (("Episodes", episodes_path), ("Frames", frames_path), ("Videos", videos_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} table not found: {path}")

    episodes_ds = _open_lance_dataset(episodes_path)
    frames_ds = _open_lance_dataset(frames_path)
    videos_ds = _open_lance_dataset(videos_path)

    print(f"Episodes schema ({episodes_path}):")
    print(episodes_ds.schema)
    episodes_versions = episodes_ds.versions()
    print(f"Episodes versions: count={len(episodes_versions)} latest={episodes_ds.latest_version}")
    print("\n---")
    print(f"Frames schema ({frames_path}):")
    print(frames_ds.schema)
    frames_versions = frames_ds.versions()
    print(f"Frames versions: count={len(frames_versions)} latest={frames_ds.latest_version}")
    print("\n---")
    print(f"Videos schema ({videos_path}):")
    print(videos_ds.schema)
    videos_versions = videos_ds.versions()
    print(f"Videos versions: count={len(videos_versions)} latest={videos_ds.latest_version}")


if __name__ == "__main__":
    main()
