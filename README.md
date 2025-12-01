# Pose Estimation Comparison

This repository compares different pose estimation algorithms (MediaPipe, YOLO-Pose, MMPose, MoveNet) on video data.

## 1. Prerequisites

### Data
Ensure you have the input video file in the `data/` directory. The scripts default to looking for:
`data/14-10-BO-0001_short.mp4`

If you have a different video, you can pass its path as an argument to the scripts.

### Tools
You need `uv` installed for environment management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Setup

Run the setup script to create isolated virtual environments for each model and install their dependencies:

```bash
chmod +x setup_all_envs.sh
./setup_all_envs.sh
```

This will create:
*   `.mediapipe/` (Python 3.9)
*   `.yolopose/` (Python 3.10)
*   `.mmpose/` (Python 3.10)
*   `.movenet/` (Python 3.10)
*   `model_weights/` (Directory for downloaded models)

## 3. Usage

### Run Comparison
To run all installed models on the default video and save results:

```bash
chmod +x compare_models.sh
./compare_models.sh
```
Results will be saved in `results/` and a summary log in `comparison_results.txt`.

### Run Individually
You can run each model individually using its dedicated virtual environment:

**MediaPipe:**
```bash
.mediapipe/bin/python video_pose_mediapipe.py [VIDEO_PATH] [OUTPUT_PATH]
```

**YOLO-Pose:**
```bash
.yolopose/bin/python video_pose_yolo.py [VIDEO_PATH] [OUTPUT_PATH]
```

**MMPose:**
```bash
.mmpose/bin/python video_pose_mmpose.py [VIDEO_PATH] [OUTPUT_PATH]
```

**MoveNet:**
```bash
.movenet/bin/python video_pose_movenet.py [VIDEO_PATH] [OUTPUT_PATH]
```

## 4. Models Implemented

*   **MediaPipe:** Uses MediaPipe Tasks API (GPU supported).
*   **YOLO-Pose:** Uses Ultralytics YOLO11n-pose.
*   **MMPose:** Uses RTMO-l via MMPoseInferencer.
*   **MoveNet:** Uses MoveNet MultiPose Lightning (TensorFlow Hub).

