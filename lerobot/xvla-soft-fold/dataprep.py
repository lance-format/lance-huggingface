#!/usr/bin/env python3
"""
Convert a LeRobot v3.0 dataset to the following Lance tables:
- frames.lance: frame-level rows from source parquet shards
- videos.lance: raw mp4 bytes (lossless) + indexing metadata
- episodes.lance: one row per episode + encoded segment blobs for each camera angle + from/to timestamps
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

TARGET_EPISODE_FPS = 20.0
MAX_BYTES_PER_FILE = 15 * 1024 * 1024 * 1024
SOURCE_FRAME_COLUMNS = [
    "timestamp",
    "action",
    "observation.state",
    "episode_index",
    "task_index",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sanitize_name(name: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]", "_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "col"
    if out[0].isdigit():
        out = f"c_{out}"
    return out


def _sanitize_column_names(column_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name in column_names:
        base = _sanitize_name(name)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        mapping[name] = candidate
        used.add(candidate)
    return mapping


def _list_camera_video_keys(features: dict[str, dict[str, Any]]) -> list[str]:
    return [k for k, ft in features.items() if ft.get("dtype") == "video"]


def _codec_candidates(codec: str) -> list[str]:
    normalized = codec.strip().lower()
    alias_map = {
        "av1": ["libaom-av1", "av1", "libsvtav1", "rav1e"],
        "h264": ["libx264", "h264"],
        "avc1": ["libx264", "h264"],
        "hevc": ["libx265", "hevc"],
        "h265": ["libx265", "hevc"],
        "vp9": ["libvpx-vp9", "vp9"],
        "vp8": ["libvpx", "vp8"],
    }
    ordered = [normalized]
    for candidate in alias_map.get(normalized, []):
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    return json.loads(info_path.read_text())


def _load_episodes(root: Path) -> pa.Table:
    ep_paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not ep_paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {root / 'meta' / 'episodes'}")
    return pq.read_table(ep_paths)


def _list_data_paths(root: Path) -> list[Path]:
    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files under {root / 'data'}")
    return data_paths


def _list_video_paths(root: Path) -> list[Path]:
    videos_root = root / "videos"
    video_paths = sorted(videos_root.glob("**/*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No video files found under {videos_root}")
    return video_paths


def _parse_prefixed_index(name: str, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", name)
    return int(m.group(1)) if m else None


def _extract_video_segment_bytes(
    video_path: Path,
    from_ts: float,
    to_ts: float,
    fps: float,
    vcodec: str,
    pix_fmt: str = "yuv420p",
) -> bytes:
    try:
        import av  # type: ignore
    except Exception as e:
        raise ImportError("'av' is required. Install with `pip install av`.") from e

    if to_ts <= from_ts:
        raise ValueError(f"Invalid segment range: from_ts={from_ts}, to_ts={to_ts}, video={video_path}")
    if fps <= 0:
        raise ValueError(f"Invalid fps={fps!r} for {video_path}")

    in_container = av.open(str(video_path), mode="r")
    if not in_container.streams.video:
        in_container.close()
        raise ValueError(f"No video stream found in {video_path}")
    v_in = in_container.streams.video[0]

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    fps_fraction = Fraction(str(fps)).limit_denominator(1000)

    v_out = None
    selected_codec = None
    add_stream_error: Exception | None = None
    for codec_candidate in _codec_candidates(vcodec):
        try:
            v_out = out.add_stream(codec_candidate, rate=fps_fraction)
            selected_codec = codec_candidate
            break
        except Exception as e:
            add_stream_error = e
    if v_out is None or selected_codec is None:
        raise RuntimeError(
            f"Failed to initialize encoder for codec={vcodec!r}; tried={_codec_candidates(vcodec)!r}"
        ) from add_stream_error

    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt
    v_out.time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
    out.start_encoding()

    start_pts = int(from_ts / float(v_in.time_base)) if v_in.time_base else 0
    in_container.seek(start_pts, any_frame=False, stream=v_in)

    frame_count = 0
    stop = False
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue
            t = float(frame.pts * frame.time_base) if frame.pts is not None else (frame_count / fps)
            if t < from_ts:
                continue
            if t >= to_ts:
                stop = True
                break

            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)
            frame_count += 1
        if stop:
            break

    for pkt in v_out.encode():
        out.mux(pkt)
    out.close()
    in_container.close()

    data = buf.getvalue()
    if not data:
        raise RuntimeError(f"Encoded segment is empty for {video_path} [{from_ts}, {to_ts})")
    return data


def _build_episode_schema(camera_keys: list[str]) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("episode_index", pa.int64(), nullable=False),
        pa.field("task_index", pa.int64(), nullable=False),
        pa.field("fps", pa.int32(), nullable=False),
        pa.field("timestamps", pa.list_(pa.float32())),
        pa.field("actions", pa.list_(pa.list_(pa.float32()))),
        pa.field("observation_state", pa.list_(pa.list_(pa.float32()))),
    ]
    for ck in camera_keys:
        pref = _sanitize_name(ck)
        fields.append(
            pa.field(
                f"{pref}_video_blob",
                pa.large_binary(),
                nullable=True,
                metadata={b"lance-encoding:blob": b"true"},
            )
        )
        fields.append(pa.field(f"{pref}_from_timestamp", pa.float64(), nullable=True))
        fields.append(pa.field(f"{pref}_to_timestamp", pa.float64(), nullable=True))
    return pa.schema(fields)


def _build_videos_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("camera_angle", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=True),
            pa.field("file_index", pa.int32(), nullable=True),
            pa.field("relative_path", pa.string(), nullable=False),
            pa.field("filename", pa.string(), nullable=False),
            pa.field("file_size_bytes", pa.int64(), nullable=False),
            pa.field("sha256", pa.string(), nullable=False),
            pa.field(
                "video_blob",
                pa.large_binary(),
                nullable=False,
                metadata={b"lance-encoding:blob": b"true"},
            ),
        ]
    )


def _prepare_output_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Output path already exists: {path} (use --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)


def _single_row_record_batch(row: dict[str, Any], schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict({k: [row.get(k)] for k in schema.names}, schema=schema)


def _write_frames_table(root: Path, out_path: Path, *, overwrite: bool, limit: int, log_every: int) -> None:
    data_paths = _list_data_paths(root)
    _prepare_output_path(out_path, overwrite=overwrite)

    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance' package; install with `pip install lance`.") from e

    first_shard = pq.read_table(data_paths[0])
    expected_source_cols = list(first_shard.column_names)
    column_map = _sanitize_column_names(expected_source_cols)
    renamed_cols = [column_map[name] for name in expected_source_cols]
    first_shard = first_shard.rename_columns(renamed_cols)
    metadata = dict(first_shard.schema.metadata or {})
    metadata[b"source-column-name-map"] = json.dumps(column_map, sort_keys=True).encode("utf-8")
    frames_schema = first_shard.replace_schema_metadata(metadata).schema

    stats = {"rows_written": 0}

    def batch_iter() -> Iterator[pa.RecordBatch]:
        wrote_any = False

        for file_idx, parquet_path in enumerate(data_paths, start=1):
            if limit > 0 and stats["rows_written"] >= limit:
                break

            shard = pq.read_table(parquet_path)
            source_cols = list(shard.column_names)
            if source_cols != expected_source_cols:
                raise RuntimeError(
                    f"Frame shard schema mismatch in {parquet_path}. "
                    "Expected same column order as first shard."
                )

            renamed_cols = [column_map[name] for name in source_cols]
            shard = shard.rename_columns(renamed_cols)

            if limit > 0:
                remaining = limit - stats["rows_written"]
                if remaining <= 0:
                    break
                if shard.num_rows > remaining:
                    shard = shard.slice(0, remaining)

            if shard.num_rows == 0:
                continue

            if wrote_any is False:
                shard = shard.replace_schema_metadata(dict(frames_schema.metadata or {}))

            for batch in shard.to_batches():
                yield batch

            wrote_any = True
            stats["rows_written"] += shard.num_rows
            if stats["rows_written"] % max(1, log_every) == 0 or file_idx == len(data_paths):
                _log(f"frames rows written: {stats['rows_written']}")

        if not wrote_any:
            raise RuntimeError("No frame rows were produced; cannot create frames.lance")

    lance.write_dataset(
        batch_iter(),
        str(out_path),
        schema=frames_schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )
    _log(f"frames.lance written: {out_path} rows={stats['rows_written']}")


def _write_videos_table(root: Path, out_path: Path, *, overwrite: bool, limit: int, log_every: int) -> None:
    video_paths = _list_video_paths(root)
    if limit > 0:
        video_paths = video_paths[:limit]
    _prepare_output_path(out_path, overwrite=overwrite)

    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance' package; install with `pip install lance`.") from e

    schema = _build_videos_schema()
    videos_root = root / "videos"

    stats = {"rows_written": 0}

    def batch_iter() -> Iterator[pa.RecordBatch]:
        if not video_paths:
            raise RuntimeError("No video rows were produced; cannot create videos.lance")

        for i, video_path in enumerate(video_paths, start=1):
            rel = video_path.relative_to(videos_root)
            rel_parts = rel.parts
            camera_angle = rel_parts[0] if rel_parts else ""
            chunk_dir = rel_parts[1] if len(rel_parts) > 1 else ""
            chunk_index = _parse_prefixed_index(chunk_dir, "chunk")
            file_index = _parse_prefixed_index(video_path.stem, "file")
            blob = video_path.read_bytes()

            row = {
                "camera_angle": camera_angle,
                "chunk_index": chunk_index,
                "file_index": file_index,
                "relative_path": rel.as_posix(),
                "filename": video_path.name,
                "file_size_bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "video_blob": blob,
            }
            yield _single_row_record_batch(row, schema)

            stats["rows_written"] += 1
            if i % max(1, log_every) == 0 or i == len(video_paths):
                _log(f"videos rows written: {i}/{len(video_paths)}")

    lance.write_dataset(
        batch_iter(),
        str(out_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )
    _log(f"videos.lance written: {out_path} rows={stats['rows_written']}")


def _iter_episode_rows(
    root: Path,
    info: dict[str, Any],
    episodes_tbl: pa.Table,
    camera_keys: list[str],
    *,
    episode_vcodec: str,
    limit: int,
    log_every: int,
) -> Iterator[dict[str, Any]]:
    source_fps = info.get("fps")
    if source_fps is not None and float(source_fps) != TARGET_EPISODE_FPS:
        _log(f"info.json fps={source_fps} but forcing episode encoding to {TARGET_EPISODE_FPS:.0f} FPS")

    fps = int(TARGET_EPISODE_FPS)
    features = info.get("features", {})
    camera_encoding: dict[str, tuple[str, str]] = {}
    for ck in camera_keys:
        camera_info = features.get(ck, {}).get("info", {}) if isinstance(features.get(ck, {}), dict) else {}
        source_codec = str(camera_info.get("video.codec") or "h264")
        source_pix_fmt = str(camera_info.get("video.pix_fmt") or "yuv420p")
        camera_encoding[ck] = (source_codec, source_pix_fmt)

    if episode_vcodec.strip().lower() == "source" and any(codec.lower() == "av1" for codec, _ in camera_encoding.values()):
        _log("Episode segment encoding is using AV1 from info.json; this can be very slow.")

    data_paths = _list_data_paths(root)
    episodes = episodes_tbl.to_pydict()
    ep_rows: dict[int, dict[str, Any]] = {}
    for i, ep_idx in enumerate(episodes["episode_index"]):
        ep_rows[int(ep_idx)] = {k: episodes[k][i] for k in episodes_tbl.column_names}

    rows_emitted = 0
    current_ep: int | None = None
    current_ts: list[float] = []
    current_actions: list[list[float]] = []
    current_obs: list[list[float]] = []
    current_task_idx = 0

    def finalize_episode(ep_idx: int) -> dict[str, Any] | None:
        nonlocal current_ts, current_actions, current_obs, rows_emitted
        if limit > 0 and rows_emitted >= limit:
            return None

        started = time.time()
        ep = ep_rows.get(ep_idx)
        if ep is None:
            raise RuntimeError(f"Episode metadata not found for episode_index={ep_idx}")

        row: dict[str, Any] = {
            "episode_index": int(ep_idx),
            "task_index": int(current_task_idx),
            "fps": fps,
            "timestamps": current_ts.copy(),
            "actions": [list(map(float, a)) for a in current_actions],
            "observation_state": [list(map(float, s)) for s in current_obs],
        }

        for ck in camera_keys:
            from_key = f"videos/{ck}/from_timestamp"
            to_key = f"videos/{ck}/to_timestamp"
            chunk_key = f"videos/{ck}/chunk_index"
            file_key = f"videos/{ck}/file_index"
            pref = _sanitize_name(ck)

            if from_key not in ep or to_key not in ep or chunk_key not in ep or file_key not in ep:
                row[f"{pref}_video_blob"] = None
                row[f"{pref}_from_timestamp"] = None
                row[f"{pref}_to_timestamp"] = None
                continue

            from_ts = float(ep[from_key])
            to_ts = float(ep[to_key])
            chunk_idx = int(ep[chunk_key])
            file_idx = int(ep[file_key])
            row[f"{pref}_from_timestamp"] = from_ts
            row[f"{pref}_to_timestamp"] = to_ts

            vpath_tpl = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            vpath = root / vpath_tpl.format(video_key=ck, chunk_index=chunk_idx, file_index=file_idx)
            source_codec, source_pix_fmt = camera_encoding[ck]
            encode_codec = source_codec if episode_vcodec.strip().lower() == "source" else episode_vcodec
            encode_pix_fmt = source_pix_fmt if episode_vcodec.strip().lower() == "source" else "yuv420p"

            row[f"{pref}_video_blob"] = _extract_video_segment_bytes(
                vpath,
                from_ts,
                to_ts,
                float(fps),
                encode_codec,
                encode_pix_fmt,
            )

        rows_emitted += 1
        if rows_emitted % max(1, log_every) == 0 or rows_emitted == 1:
            _log(
                f"episodes finalized: {rows_emitted} latest_episode_index={ep_idx} "
                f"seq_len={len(current_ts)} elapsed={time.time() - started:.1f}s"
            )
        return row

    for shard_idx, shard_path in enumerate(data_paths, start=1):
        if limit > 0 and rows_emitted >= limit:
            break
        _log(f"reading episode shard {shard_idx}/{len(data_paths)}: {shard_path.name}")
        t = pq.read_table(shard_path, columns=SOURCE_FRAME_COLUMNS)
        d = t.to_pydict()
        for i in range(t.num_rows):
            ep = int(d["episode_index"][i])
            if current_ep is None:
                current_ep = ep
            elif ep != current_ep:
                row = finalize_episode(current_ep)
                if row is not None:
                    yield row
                if limit > 0 and rows_emitted >= limit:
                    return
                current_ep = ep
                current_ts = []
                current_actions = []
                current_obs = []

            ts = float(d["timestamp"][i]) if d["timestamp"][i] is not None else 0.0
            current_ts.append(ts)

            act = d["action"][i]
            if act is not None:
                current_actions.append([float(x) for x in act])

            obs = d["observation.state"][i]
            if obs is not None:
                current_obs.append([float(x) for x in obs])

            task = d["task_index"][i]
            current_task_idx = int(task) if task is not None else 0

    if current_ep is not None and (limit == 0 or rows_emitted < limit):
        row = finalize_episode(current_ep)
        if row is not None:
            yield row


def _write_episodes_table(
    root: Path,
    out_path: Path,
    *,
    info: dict[str, Any],
    overwrite: bool,
    limit: int,
    log_every: int,
    episode_vcodec: str,
) -> None:
    episodes_tbl = _load_episodes(root)
    camera_keys = _list_camera_video_keys(info["features"])
    if not camera_keys:
        raise RuntimeError("No video camera keys found in info features")

    schema = _build_episode_schema(camera_keys)
    _prepare_output_path(out_path, overwrite=overwrite)

    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance' package; install with `pip install lance`.") from e

    stats = {"rows_written": 0}

    def batch_iter() -> Iterator[pa.RecordBatch]:
        wrote_any = False
        for row in _iter_episode_rows(
            root,
            info,
            episodes_tbl,
            camera_keys,
            episode_vcodec=episode_vcodec,
            limit=limit,
            log_every=log_every,
        ):
            yield _single_row_record_batch(row, schema)
            wrote_any = True
            stats["rows_written"] += 1
        if not wrote_any:
            raise RuntimeError("No episode rows were produced; cannot create episodes.lance")

    lance.write_dataset(
        batch_iter(),
        str(out_path),
        schema=schema,
        mode="create",
        max_bytes_per_file=MAX_BYTES_PER_FILE,
    )
    _log(f"episodes.lance written: {out_path} rows={stats['rows_written']}")


def convert_dataset_v30_to_lance_bundle(
    root: Path,
    out_root: Path,
    *,
    overwrite: bool,
    limit: int,
    log_every: int,
    episode_vcodec: str,
) -> None:
    info = _load_info(root)
    out_root.mkdir(parents=True, exist_ok=True)

    frames_path = out_root / "frames.lance"
    videos_path = out_root / "videos.lance"
    episodes_path = out_root / "episodes.lance"
    out_paths = [frames_path, videos_path, episodes_path]

    for path in out_paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output path already exists: {path} (use --overwrite)")
    if overwrite:
        for path in out_paths:
            if path.exists():
                shutil.rmtree(path)

    # Requested order: write lightweight tables first, expensive episodes last.
    _write_frames_table(root, frames_path, overwrite=False, limit=limit, log_every=log_every)
    _write_videos_table(root, videos_path, overwrite=False, limit=limit, log_every=log_every)
    _write_episodes_table(
        root,
        episodes_path,
        info=info,
        overwrite=False,
        limit=limit,
        log_every=log_every,
        episode_vcodec=episode_vcodec,
    )
    _log(f"Lance bundle written under: {out_root}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert v3.0 dataset to three Lance tables: frames, videos, episodes"
    )
    p.add_argument("--root", type=str, required=True, help="Local v3.0 dataset root")
    p.add_argument(
        "--out",
        type=str,
        required=False,
        help="Output directory containing episodes.lance/frames.lance/videos.lance (default: --root)",
    )
    p.add_argument("--overwrite", action="store_true", help="Delete outputs before writing")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Hard row/item limit for each table. 0 means no limit.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Progress log frequency",
    )
    p.add_argument(
        "--episode-vcodec",
        type=str,
        default="libx264",
        help="Episode segment encoder. Use 'source' to use per-camera video.codec from info.json.",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")

    out_root = Path(args.out).resolve() if args.out else root
    convert_dataset_v30_to_lance_bundle(
        root,
        out_root,
        overwrite=args.overwrite,
        limit=args.limit,
        log_every=args.log_every,
        episode_vcodec=args.episode_vcodec,
    )


if __name__ == "__main__":
    main()
