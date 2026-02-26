---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

  **Repository:** [X-VLA](https://thu-air-dream.github.io/X-VLA/)

  **License:** Apache 2.0

  **Paper:** *Zheng et al., 2025, “X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model”* ([arXiv:2510.10274](https://arxiv.org/pdf/2510.10274))


## What this dataset contains

This is the Lance-format version of [lerobot/xvla-soft-fold](https://huggingface.co/datasets/lerobot/xvla-soft-fold), designed for efficient frame-level sampling and sequential episode loading.

- `1,542` episodes
- `2,852,512` frames
- `20` FPS
- 3 camera streams per episode (`cam_high`, `cam_left_wrist`, `cam_right_wrist`)
- robot state vectors and action vectors aligned to frame timestamps

## Metadata (`meta/info.json`)

The original dataset metadata is preserved below for reference:

```json
{
    "codebase_version": "v3.0",
    "robot_type": "franka",
    "total_episodes": 1542,
    "total_frames": 2852512,
    "total_tasks": 1,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 500,
    "fps": 20,
    "splits": {
        "train": "0:1542"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                96
            ],
            "names": [
                "eef_euler_0",
                "eef_euler_1",
                "eef_euler_2",
                "eef_euler_3",
                "eef_euler_4",
                "eef_euler_5",
                "eef_euler_6",
                "eef_euler_7",
                "eef_euler_8",
                "eef_euler_9",
                "eef_euler_10",
                "eef_euler_11",
                "eef_euler_12",
                "eef_euler_13",
                "eef_quat_0",
                "eef_quat_1",
                "eef_quat_2",
                "eef_quat_3",
                "eef_quat_4",
                "eef_quat_5",
                "eef_quat_6",
                "eef_quat_7",
                "eef_quat_8",
                "eef_quat_9",
                "eef_quat_10",
                "eef_quat_11",
                "eef_quat_12",
                "eef_quat_13",
                "eef_quat_14",
                "eef_quat_15",
                "eef6d_0",
                "eef6d_1",
                "eef6d_2",
                "eef6d_3",
                "eef6d_4",
                "eef6d_5",
                "eef6d_6",
                "eef6d_7",
                "eef6d_8",
                "eef6d_9",
                "eef6d_10",
                "eef6d_11",
                "eef6d_12",
                "eef6d_13",
                "eef6d_14",
                "eef6d_15",
                "eef6d_16",
                "eef6d_17",
                "eef6d_18",
                "eef6d_19",
                "eef_left_time",
                "eef_right_time",
                "qpos_0",
                "qpos_1",
                "qpos_2",
                "qpos_3",
                "qpos_4",
                "qpos_5",
                "qpos_6",
                "qpos_7",
                "qpos_8",
                "qpos_9",
                "qpos_10",
                "qpos_11",
                "qpos_12",
                "qpos_13",
                "qvel_0",
                "qvel_1",
                "qvel_2",
                "qvel_3",
                "qvel_4",
                "qvel_5",
                "qvel_6",
                "qvel_7",
                "qvel_8",
                "qvel_9",
                "qvel_10",
                "qvel_11",
                "qvel_12",
                "qvel_13",
                "effort_0",
                "effort_1",
                "effort_2",
                "effort_3",
                "effort_4",
                "effort_5",
                "effort_6",
                "effort_7",
                "effort_8",
                "effort_9",
                "effort_10",
                "effort_11",
                "effort_12",
                "effort_13",
                "qpos_left_time",
                "qpos_right_time"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": {
                "motors": [
                    "joint_action_0",
                    "joint_action_1",
                    "joint_action_2",
                    "joint_action_3",
                    "joint_action_4",
                    "joint_action_5",
                    "joint_action_6",
                    "joint_action_7",
                    "joint_action_8",
                    "joint_action_9",
                    "joint_action_10",
                    "joint_action_11",
                    "joint_action_12",
                    "joint_action_13"
                ]
            }
        },
        "time_stamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": {
                "values": [
                    "global_timestamp"
                ]
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```

## Dataset structure

The dataset is organized under `data/` with three Lance tables:

### Frames table

This is the main table for model training and analytics at frame granularity. Each row is one frame with aligned state/action metadata and indexing fields so you can filter by episode, iterate temporally, or build sampled batches directly.

Schema:
- `observation_state` (`list<float>`): robot state vector for that frame.
- `action` (`list<float>`): action vector for that frame.
- `time_stamp` (`float`): original source timestamp field.
- `timestamp` (`float`): canonical frame timestamp.
- `frame_index` (`int64`): frame index within episode.
- `episode_index` (`int64`): parent episode id.
- `index` (`int64`): global frame index.
- `task_index` (`int64`): task id.

### Episodes table

This table is optimized for sequence-aware loading. Each row represents one complete episode and stores per-episode arrays (`timestamps`, `actions`, `observation_state`) plus per-camera video blobs and timestamp ranges. Use this table when you need contiguous windows, trajectory-level batching, or synchronized decoding from episode-level video chunks.

Schema:
- `episode_index` (`int64`, required): episode id.
- `task_index` (`int64`, required): task id.
- `fps` (`int32`, required): frame rate.
- `timestamps` (`list<float>`): per-frame timestamps for the episode.
- `actions` (`list<list<float>>`): per-frame action vectors.
- `observation_state` (`list<list<float>>`): per-frame robot state vectors.
- `observation_images_cam_high_video_blob` (`large_binary` blob): encoded video segment for `cam_high`.
- `observation_images_cam_high_from_timestamp` (`double`): segment start time for `cam_high`.
- `observation_images_cam_high_to_timestamp` (`double`): segment end time for `cam_high`.
- `observation_images_cam_left_wrist_video_blob` (`large_binary` blob): encoded video segment for `cam_left_wrist`.
- `observation_images_cam_left_wrist_from_timestamp` (`double`): segment start time for `cam_left_wrist`.
- `observation_images_cam_left_wrist_to_timestamp` (`double`): segment end time for `cam_left_wrist`.
- `observation_images_cam_right_wrist_video_blob` (`large_binary` blob): encoded video segment for `cam_right_wrist`.
- `observation_images_cam_right_wrist_from_timestamp` (`double`): segment start time for `cam_right_wrist`.
- `observation_images_cam_right_wrist_to_timestamp` (`double`): segment end time for `cam_right_wrist`.

### Videos table

This table stores raw MP4 payloads from the source and file-level provenance metadata. It is useful when you want direct access to original encoded video assets, integrity checks (`sha256`), or custom decoding pipelines that operate on the original video files themselves, rather than episode/frame abstractions.

Schema:
- `camera_angle` (`string`, required): camera key.
- `chunk_index` (`int32`): chunk id parsed from path.
- `file_index` (`int32`): file id parsed from path.
- `relative_path` (`string`, required): original relative path in dataset.
- `filename` (`string`, required): MP4 filename.
- `file_size_bytes` (`int64`, required): file size.
- `sha256` (`string`, required): SHA256 digest.
- `video_blob` (`large_binary`, required blob): raw MP4 bytes.

## Usage

In the following sections, we'll show how to work with the dataset in Lance or LanceDB.

### Read with Lance

```python
import lance

root_path = "hf://datasets/lance-format/lerobot-xvla-soft-fold/data"
frames_table_name = "frames.lance"
episodes_table_name = "episodes.lance"
videos_table_name = "videos.lance"

ds = lance.dataset(f"{root_path}/{frames_table_name}")
print(ds.count_rows())

ds = lance.dataset(f"{root_path}/{episodes_table_name}")
print(ds.count_rows())

ds = lance.dataset(f"{root_path}/{videos_table_name}")
print(ds.count_rows())

# Returns:
# 2852512
# 1542
# 104
```

### Inspect a few frames

```python
import lance

root_path = "hf://datasets/lance-format/lerobot-xvla-soft-fold/data"
frames_table_name = "frames.lance"

frames = lance.dataset(f"{root_path}/{frames_table_name}")
print(f"There are {frames.count_rows()} frames in total")

# pip install polars
res = frames.scanner(
    columns=["episode_index", "frame_index", "timestamp"],
    limit=2,
).to_table()
print(res)

# Returns
# There are 2852512 frames in total
# pyarrow.Table
# episode_index: int64
# frame_index: int64
# timestamp: float
# ----
# episode_index: [[0,0]]
# frame_index: [[0,1]]
# timestamp: [[0,0.05]]
```

### Retrieving and saving video blobs

```py
from pathlib import Path
import lance

root_path = "hf://datasets/lance-format/lerobot-xvla-soft-fold/data"
episodes_table_name = "episodes.lance"
ds = lance.dataset(f"{root_path}/{episodes_table_name}")

out = Path("video_blobs")
out.mkdir(exist_ok=True)

# Retrieve first two videos from the episodes table
for offset in range(0, 2):
    row = (
        ds.scanner(
            columns=["episode_index", "observation_images_cam_high_video_blob"],
            blob_handling="all_binary",
            limit=2,
            offset=offset,
        )
        .to_table()
        .to_pylist()[0]
    )
    # Write the video blob to a file
    (out / f"episode_{row['episode_index']}.mp4").write_bytes(
        row["observation_images_cam_high_video_blob"]
    )
```
This outputs the retrieved blobs as MP4 files in a local directory.

### Random seek on subsets of video

The snippet shown below reads one episode’s video blob directly from HF Hub via Lance, computes a tiny time window inside that episode, opens the blob as a stream (without downloading full data into a local file), seeks to the start timestamp, and prints the blob size plus the exact seek positions in seconds and stream PTS units.

```py
import av
import lance

DATASET_URI = "hf://datasets/lance-format/lerobot-xvla-soft-fold/data/episodes.lance"
EPISODE_INDEX = 30
START_OFFSET_S = 1.0
WINDOW_S = 0.5

ds = lance.dataset(DATASET_URI)
row = ds.scanner(
    columns=[
        "episode_index",
        "observation_images_cam_high_from_timestamp",
        "observation_images_cam_high_to_timestamp",
        "_rowid",
    ],
    with_row_id=True,
    filter=f"episode_index = {EPISODE_INDEX}",
    limit=1,
).to_table().to_pylist()[0]

start_s = row["observation_images_cam_high_from_timestamp"] + START_OFFSET_S
end_s = min(
    start_s + WINDOW_S,
    row["observation_images_cam_high_to_timestamp"],
)

blob = ds.take_blobs("observation_images_cam_high_video_blob", ids=[row["_rowid"]])[0]
with av.open(blob) as container:
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"

    start_pts = int(start_s / stream.time_base)
    end_pts = int(end_s / stream.time_base)
    container.seek(start_pts, stream=stream)

    print(f"episode_index={row['episode_index']}")
    print(f"blob_size_bytes={blob.size()}")
    print(f"seek_start_seconds={start_s:.3f}")
    print(f"seek_end_seconds={end_s:.3f}")
    print(f"seek_start_pts={start_pts}")
    print(f"seek_end_pts={end_pts}")

blob.close()
```

### LanceDB search

LanceDB users can also interface with the Lance dataset on the Hub. The key step is to
connect to the dataset repo and open the relevant table.

```py
import lancedb

db = lancedb.connect("hf://datasets/lance-format/lerobot-xvla-soft-fold/data")
tbl = db.open_table("episodes")

# Search without any parameters
results = (
    tbl.search()
    .select(
        [
            "episode_index",
            "observation_images_cam_high_from_timestamp",
            "observation_images_cam_high_to_timestamp",
        ]
    )
    .limit(3)
    .to_list()
)

for result in results:
    print(
        f"{result['episode_index']} | {result['observation_images_cam_high_from_timestamp']} | {result['observation_images_cam_high_to_timestamp']}"
    )

# Returns:
# 0 | 0.0 | 122.95
# 1 | 122.95 | 230.65
# 2 | 230.65 | 340.0
```

### Download

If you need to make modifications to the data or work with the raw files directly, you can do a 
full download of the dataset locally.

> **⚠️ Large dataset download**
> The full dataset is >50GB in size, so ensure you have sufficient disk space available.

```bash
uv run hf download lance-format/lerobot-xvla-soft-fold --repo-type dataset --local-dir .
```