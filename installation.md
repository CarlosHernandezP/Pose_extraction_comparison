# Pose Estimation Algorithms - Installation & Usage Guide

This guide provides instructions for setting up and running different pose estimation models using isolated virtual environments.

## Common Setup

All scripts expect the input video to be located at:
`data/14-10-BO-0001_short.mp4`

Ensure this file exists before running any script.

---

## 1. MediaPipe (GPU)

**Status:** ✅ GPU Supported (via Tasks API)

### Installation
```bash
# Create virtual environment
uv venv .mediapipe --python 3.9

# Activate (optional if using explicit path)
source .mediapipe/bin/activate

# Install dependencies
uv pip install mediapipe opencv-python tqdm --active

# Download Model File
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task -O model_weights/pose_landmarker_full.task
```

### Usage
Run the script using the dedicated environment:
```bash
.mediapipe/bin/python video_pose_mediapipe.py
```

---

## 2. YOLO-Pose (Ultralytics)

**Status:** ✅ GPU Supported 

### Installation
```bash
# Create virtual environment
uv venv .yolopose --python 3.10

# Activate
source .yolopose/bin/activate

# Install dependencies (Ultralytics includes PyTorch)
uv pip install ultralytics opencv-python tqdm --active

# (Optional) Pre-download model to model_weights/
# The script will handle download and move if missing, 
# but you can manually place 'yolo11n-pose.pt' in model_weights/
```

### Usage
Run the script using the dedicated environment:
```bash
.yolopose/bin/python video_pose_yolo.py
```

---

## 3. MMPose with RTMO

Referring to the official MMPose documentation
https://mmpose.readthedocs.io/en/latest/installation.html

**Status:** ✅ Verified (RTMO-l)    

### Installation
```bash
# Create virtual environment
uv venv .mmpose --python 3.10

# Activate
source .mmpose/bin/activate

# Install base dependencies
# Note: MMPose 1.x requires compatible PyTorch/MMCV versions.
# We found stability with PyTorch 2.1.0 and MMCV 2.1.0
uv pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121 --active
uv pip install opencv-python tqdm --active

# Install OpenMIM (MMLab package manager)
uv pip install -U openmim --active

# Install MMEngine and MMCV via MIM
# We strictly specify versions to ensure compatibility:
# mmpose 1.x <=> mmcv 2.x (<2.2.0) <=> mmdet 3.x (<3.3.0)
.mmpose/bin/mim install mmengine
# Force MMCV installation for CUDA 12.1 / Torch 2.1.0
.mmpose/bin/mim install "mmcv==2.1.0"

# Install MMDetection and MMPose
.mmpose/bin/mim install "mmdet>=3.1.0,<3.3.0"
.mmpose/bin/mim install "mmpose>=1.1.0"

# Download Config and Checkpoint
# RTMO config is not yet standard in mim, so we download it manually:
.mmpose/bin/mim download mmpose --config rtmo-l_16xb16-600e_coco-640x640 --dest configs
mv configs/*.pth model_weights/
```

### Usage
Run the script using the dedicated environment:
```bash
.mmpose/bin/python video_pose_mmpose.py
```
---

## 4. MoveNet (TensorFlow)

**Status:** ✅ Verified

### Installation
```bash
# Create virtual environment
uv venv .movenet --python 3.10

# Activate
source .movenet/bin/activate

# Install dependencies
uv pip install tensorflow tensorflow-hub opencv-python tqdm --active
```

### Usage
Run the script using the dedicated environment:
```bash
.movenet/bin/python video_pose_movenet.py
```
*(Uses MultiPose Lightning model for multi-person detection)*
