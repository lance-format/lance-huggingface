#!/usr/bin/env python3
"""
Example: use Lance tables (episodes + frames) for random frame and window sampling.

Prerequisites:
    pip install lance av pyarrow torch

Usage:
    uv run use_lance_dataset.py \
        --episodes /path/to/<repo>.episodes.lance \
        --frames /path/to/<repo>.frames.lance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow as pa

try:
    import torch

    _TorchDatasetBase = torch.utils.data.Dataset
except Exception:
    torch = None  # type: ignore[assignment]

    class _TorchDatasetBase:  # type: ignore[no-redef]
        pass


def _open_lance_dataset(path: Path, storage_options: Optional[Dict[str, str]] = None):
    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance'. Install with `pip install lance`.") from e

    try:
        return lance.dataset(str(path), storage_options=storage_options)
    except TypeError:
        return lance.dataset(str(path))


class LanceFrameDataset(_TorchDatasetBase):
    """
    Frame-wise dataset wrapper on a per-episode Lance table.

    Returns frame-level items with keys:
    'timestamp', 'frame_index', 'episode_index', 'index', 'task_index',
    and optional 'action', 'observation.state', plus camera frame keys.
    """

    def __init__(
        self,
        lance_path: str | Path,
        image_transforms: Optional[Callable] = None,
        tolerance_s: float = 1e-4,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        if torch is None:
            raise ImportError(
                "Missing 'torch' dependency; install with `pip install torch`."
            )
        super().__init__()

        self.lance_path = str(lance_path)
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s

        # Dependency check
        try:
            import lance  # type: ignore  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Missing 'lance' dependency; unable to read Lance table. "
                "Please `pip install lance`."
            ) from e

        self._ds = _open_lance_dataset(Path(self.lance_path), storage_options=storage_options)
        self._schema: pa.Schema = self._ds.schema

        # Detect episode-level video blob columns:
        # - preferred: video_<camera_key>
        # - repo schema: <camera_key>_video_blob
        self._video_cols: List[str] = []
        for name in self._schema.names:
            field = self._schema.field(name)
            meta = field.metadata or {}
            is_blob_meta = meta.get(b"lance-encoding:blob") == b"true"
            is_name_match = name.startswith("video_") or name.endswith("_video_blob")
            if is_name_match and is_blob_meta:
                self._video_cols.append(name)

        self._camera_keys: List[str] = [self._denorm_cam_col(c) for c in self._video_cols]

        # Build global frame-index mapping.
        if self._schema.get_field_index("length") != -1:
            lengths_tbl = self._ds.scanner(columns=["length"]).to_table()
            lengths: List[int] = [
                int(x) if x is not None else 0 for x in lengths_tbl.column("length").to_pylist()
            ]
        else:
            timestamps_tbl = self._ds.scanner(columns=["timestamps"]).to_table()
            ts_lists = timestamps_tbl.column("timestamps").to_pylist()
            lengths = [len(ts) if ts is not None else 0 for ts in ts_lists]

        self._episode_lengths = lengths
        self._prefix: List[int] = []
        total = 0
        for episode_len in self._episode_lengths:
            self._prefix.append(total)
            total += episode_len
        self._total_frames = total

        # Simple expansion map for O(1) lookup in __getitem__.
        self._global_map: List[Tuple[int, int]] = []
        for ep_row, episode_len in enumerate(self._episode_lengths):
            for frame_local in range(episode_len):
                self._global_map.append((ep_row, frame_local))

    @staticmethod
    def _denorm_cam_col(col_name: str) -> str:
        """Restore camera column names to dot notation when possible."""
        if col_name.startswith("video_"):
            return col_name[len("video_") :].replace("_", ".")
        if col_name.endswith("_video_blob"):
            return col_name[: -len("_video_blob")].replace("_", ".")
        return col_name

    def __len__(self) -> int:
        return self._total_frames

    def _get_episode_row(self, row_idx: int) -> Dict[str, Any]:
        # Read a single episode row via take([row_idx]) to avoid full-table load.
        row_tbl = self._ds.take([row_idx])

        def _get_scalar(name: str, default: Any = None) -> Any:
            if name not in row_tbl.column_names:
                return default
            value = row_tbl.column(name)[0].as_py()
            return default if value is None else value

        def _get_list(name: str) -> Any:
            if name not in row_tbl.column_names:
                return None
            return row_tbl.column(name)[0].as_py()

        obs_col = "obs_state" if "obs_state" in row_tbl.column_names else "observation_state"
        length_val = _get_scalar("length", self._episode_lengths[row_idx])

        return {
            "episode_index": int(_get_scalar("episode_index", row_idx)),
            "task_index": int(_get_scalar("task_index", 0)),
            "fps": int(_get_scalar("fps", 0)),
            "length": int(length_val if length_val is not None else 0),
            "timestamps": _get_list("timestamps") or [],
            "actions": _get_list("actions"),
            "obs_state": _get_list(obs_col),
        }

    def _decode_single_frame(self, camera_col: str, ep_row_idx: int, ts: float):
        """
        Decode a single frame at timestamp from episode blob.

        Returns torch.Tensor(C,H,W,float32) or None.
        """
        if self.image_transforms is None:
            return None

        try:
            import av  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing 'av' dependency; unable to decode video; please `pip install av`."
            ) from e

        blobs = self._ds.take_blobs(camera_col, ids=[ep_row_idx])
        if not blobs:
            return None

        blob = blobs[0]
        if blob is None:
            return None

        chosen = None
        with av.open(blob, mode="r") as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]

            if stream.time_base:
                start_pts = int(ts / float(stream.time_base))
                try:
                    container.seek(start_pts, any_frame=False, stream=stream)
                except Exception:
                    pass

            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame is None:
                        continue
                    t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                    if t >= ts - self.tolerance_s:
                        chosen = frame
                        break
                if chosen is not None:
                    break

        if chosen is None:
            return None

        try:
            import numpy as np

            arr = chosen.to_ndarray(format="rgb24")
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        except Exception:
            return None

        try:
            tensor = self.image_transforms(tensor)
        except Exception:
            pass
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += self._total_frames
        if idx < 0 or idx >= self._total_frames:
            raise IndexError(f"Index out of range: {idx} / {self._total_frames}")

        ep_row_idx, frame_local = self._global_map[idx]
        ep = self._get_episode_row(ep_row_idx)
        timestamps = ep["timestamps"] or []

        if frame_local < len(timestamps):
            ts_val = float(timestamps[frame_local])
        else:
            ts_val = float(frame_local) / float(max(1, ep["fps"]))

        item: Dict[str, Any] = {
            "timestamp": torch.tensor([ts_val], dtype=torch.float32),
            "frame_index": torch.tensor([frame_local], dtype=torch.int64),
            "episode_index": torch.tensor([ep["episode_index"]], dtype=torch.int64),
            "index": torch.tensor([idx], dtype=torch.int64),
            "task_index": torch.tensor([ep["task_index"]], dtype=torch.int64),
        }

        actions = ep["actions"]
        if actions is not None and frame_local < len(actions):
            item["action"] = torch.tensor(actions[frame_local], dtype=torch.float32)

        obs_state = ep["obs_state"]
        if obs_state is not None and frame_local < len(obs_state):
            item["observation.state"] = torch.tensor(obs_state[frame_local], dtype=torch.float32)

        if self.image_transforms is not None and self._video_cols:
            for cam_col, cam_key in zip(self._video_cols, self._camera_keys):
                frame = self._decode_single_frame(cam_col, ep_row_idx, ts_val)
                if frame is not None:
                    item[cam_key] = frame

        return item

    def take(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Return items for given frame-level indices."""
        return [self.__getitem__(i) for i in ids]

    def sample_window(self, start: int, size: int) -> List[Dict[str, Any]]:
        """Sample a contiguous frame window."""
        if size <= 0:
            raise ValueError("Window size must be > 0")
        if start < 0 or start + size > len(self):
            raise ValueError(
                f"Window [{start}, {start + size}) is out of range for total_frames={len(self)}"
            )
        return self.take(list(range(start, start + size)))


def _parse_storage_options(raw: Optional[str]) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--storage-options must be a JSON object")
    return {str(k): str(v) for k, v in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Example: iterate frames via Lance episodes table")
    parser.add_argument("--episodes", type=str, required=True, help="Episodes Lance directory")
    parser.add_argument("--frames", type=str, required=True, help="Frames Lance directory")
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON object passed to lance.dataset(...), e.g. \'{"region":"us-east-1"}\'',
    )
    args = parser.parse_args()

    episodes_path = Path(args.episodes).resolve()
    frames_path = Path(args.frames).resolve()
    storage_options = _parse_storage_options(args.storage_options)

    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes table not found: {episodes_path}")
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames table not found: {frames_path}")

    frames_ds = _open_lance_dataset(frames_path, storage_options=storage_options)

    ds = LanceFrameDataset(lance_path=episodes_path, storage_options=storage_options)
    print(f"Total frame samples (sum of episode lengths): {len(ds)}")
    print(f"Total frames in frames.lance (row count): {frames_ds.count_rows()}")

    for i in range(min(3, len(ds))):
        item = ds[i]
        keys = list(item.keys())
        print(f"Sample {i}: keys={keys}")

    if len(ds) > 15:
        window = ds.sample_window(10, 5)
        print(
            f"Window length: {len(window)}; "
            f"start frame_index={window[0]['frame_index'].item()} "
            f"end frame_index={window[-1]['frame_index'].item()}"
        )


if __name__ == "__main__":
    main()
