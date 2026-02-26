---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

  **Repository:** [X-VLA](https://thu-air-dream.github.io/X-VLA/)

  **License:** Apache 2.0

  **Paper:** *Zheng et al., 2025, “X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model”* ([arXiv:2510.10274](https://arxiv.org/pdf/2510.10274))


## Source data

The raw data with MP4 video files is available [here](https://huggingface.co/datasets/lerobot/xvla-soft-fold) on the Hugging Face Hub. The source data consists of large MP4 video files (~500 MB each) and metadata tables in Parquet and JSON format. Our goal is to convert them to Lance format.

It's assumed that the HF CLI is used to download the dataset locally before running the `dataprep.py` script that converts the original data to Lance format.

```bash
uv run hf download lerobot/xvla-soft-fold --repo-type dataset --local-dir .
```

This will download the entire dataset (~53 GB) including the video files, which occupy the lion's share of the storage space.

## Dataset Structure

[meta/info.json](meta/info.json):
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

## Usage

### 1) Download source dataset

```bash
# cd to this directory
cd lerobot/xvla-soft-fold

# Download the data (Parquet + JSON + MP4 files)
uv run hf download lerobot/xvla-soft-fold --repo-type dataset --local-dir .
```

### 2) Convert source data to Lance tables (`dataprep.py`)

```bash
uv run python dataprep.py \
  --root ./ \
  --out ./lance \
  --overwrite
```

This writes three Lance datasets under `./lance`:
1. `frames.lance`: Frame-level metadata from source Parquet shards.
2. `videos.lance`: Original MP4 files stored as blob rows with indexing metadata.
3. `episodes.lance`: One row per episode with per-episode arrays and segment video blobs.

### 3) Lance tables and schemas

The conversion writes three Lance datasets under `./lance`, each with a different granularity:

`episodes.lance` (one row per episode):
- `episode_index` (`int64`, required): Episode id.
- `task_index` (`int64`, required): Task id.
- `fps` (`int32`, required): Frame rate.
- `timestamps` (`list<float>`): Per-frame timestamps for the episode.
- `actions` (`list<list<float>>`): Per-frame action vectors.
- `observation_state` (`list<list<float>>`): Per-frame robot state vectors.
- `observation_images_cam_high_video_blob` (`large_binary` blob): Encoded video segment for `cam_high`.
- `observation_images_cam_high_from_timestamp` (`double`): Segment start time.
- `observation_images_cam_high_to_timestamp` (`double`): Segment end time.
- `observation_images_cam_left_wrist_video_blob` (`large_binary` blob): Encoded video segment for `cam_left_wrist`.
- `observation_images_cam_left_wrist_from_timestamp` (`double`): Segment start time.
- `observation_images_cam_left_wrist_to_timestamp` (`double`): Segment end time.
- `observation_images_cam_right_wrist_video_blob` (`large_binary` blob): Encoded video segment for `cam_right_wrist`.
- `observation_images_cam_right_wrist_from_timestamp` (`double`): Segment start time.
- `observation_images_cam_right_wrist_to_timestamp` (`double`): Segment end time.

`frames.lance` (one row per frame):
- `observation_state` (`list<float>`): Robot state vector for the frame.
- `action` (`list<float>`): Action vector for the frame.
- `time_stamp` (`float`): Source timestamp field from input Parquet.
- `timestamp` (`float`): Canonical frame timestamp used by LeRobot loaders.
- `frame_index` (`int64`): Frame index within the episode.
- `episode_index` (`int64`): Parent episode id.
- `index` (`int64`): Global frame index.
- `task_index` (`int64`): Task id.

`videos.lance` (one row per source MP4 file):
- `camera_angle` (`string`, required): Camera key.
- `chunk_index` (`int32`): Chunk id parsed from path.
- `file_index` (`int32`): File id parsed from path.
- `relative_path` (`string`, required): Original relative path in dataset.
- `filename` (`string`, required): MP4 file name.
- `file_size_bytes` (`int64`, required): File size.
- `sha256` (`string`, required): SHA256 digest.
- `video_blob` (`large_binary`, required blob): Raw MP4 bytes.

### 4) Inspect table schemas (`debug.py`)

`debug.py` is a simple script that prints the schema of each Lance table (`episodes`, `frames`, `videos`).
It is intentionally minimal so you can easily customize it to query tables, inspect selected columns, or test filters/scanners.

```bash
# Uses defaults:
#   ./lance/episodes.lance
#   ./lance/frames.lance
#   ./lance/videos.lance
uv run python debug.py
```

```bash
# Optional explicit table paths
uv run python debug.py \
  --episodes ./lance/episodes.lance \
  --frames ./lance/frames.lance \
  --videos ./lance/videos.lance
```

### 5) Frame-level sampling example (`use_lance_dataset.py`)

`use_lance_dataset.py` wraps `episodes.lance` in a `LanceFrameDataset` (`torch.utils.data.Dataset`) and prints a few sample items.
It also prints total frame count from `frames.lance`.

```bash
uv run python use_lance_dataset.py \
  --episodes ./lance/episodes.lance \
  --frames ./lance/frames.lance
```

### 6) Upload converted Lance dataset to Hugging Face

This uploads the local `./lance` folder under a top-level `data/` directory in the Hub dataset repo:

```bash
uv run hf upload \
  lance-format/lerobot-xvla-soft-fold \
  ./lance \
  data \
  --repo-type dataset
```

## Citation

If you find this dataset helpful to your project, please kindly cite its original authors.

**BibTeX:**

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```
