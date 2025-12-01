#!/bin/bash

# Exit on error
set -e

echo "================================================="
echo "  Pose Estimation Environment Setup Script"
echo "================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first."
    echo "Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create model_weights directory if it doesn't exist
if [ ! -d "model_weights" ]; then
    echo "Creating model_weights directory..."
    mkdir -p model_weights
fi

# -------------------------------------------------------
# 1. MediaPipe (GPU)
# -------------------------------------------------------
echo ""
echo "-------------------------------------------------"
echo "1. Setting up MediaPipe environment (.mediapipe)"
echo "-------------------------------------------------"
if [ -d ".mediapipe" ]; then
    echo "Environment .mediapipe already exists. Skipping creation."
else
    echo "Creating venv .mediapipe (Python 3.9)..."
    uv venv .mediapipe --python 3.9
fi

echo "Installing MediaPipe dependencies..."
source .mediapipe/bin/activate
uv pip install mediapipe opencv-python tqdm --python .mediapipe/bin/python
deactivate

# Download MediaPipe Model if missing
if [ ! -f "model_weights/pose_landmarker_full.task" ]; then
    echo "Downloading MediaPipe Pose Landmarker model..."
    wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task -O model_weights/pose_landmarker_full.task
else
    echo "MediaPipe model already exists in model_weights/."
fi


# -------------------------------------------------------
# 2. YOLO-Pose (Ultralytics)
# -------------------------------------------------------
echo ""
echo "-------------------------------------------------"
echo "2. Setting up YOLO-Pose environment (.yolopose)"
echo "-------------------------------------------------"
if [ -d ".yolopose" ]; then
    echo "Environment .yolopose already exists. Skipping creation."
else
    echo "Creating venv .yolopose (Python 3.10)..."
    uv venv .yolopose --python 3.10
fi

echo "Installing YOLO-Pose dependencies..."
source .yolopose/bin/activate
uv pip install ultralytics opencv-python tqdm --python .yolopose/bin/python
deactivate


# -------------------------------------------------------
# 3. MMPose with RTMO
# -------------------------------------------------------
echo ""
echo "-------------------------------------------------"
echo "3. Setting up MMPose environment (.mmpose)"
echo "-------------------------------------------------"
if [ -d ".mmpose" ]; then
    echo "Environment .mmpose already exists. Skipping creation."
else
    echo "Creating venv .mmpose (Python 3.10)..."
    uv venv .mmpose --python 3.10
fi

echo "Installing MMPose dependencies..."
source .mmpose/bin/activate
# Install specific PyTorch version compatible with pre-built MMCV wheels
uv pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121 --python .mmpose/bin/python --active
uv pip install opencv-python tqdm openmim --python .mmpose/bin/python --active

# MMLab deps via mim
# Use the mim executable installed in the venv
MIM_CMD=".mmpose/bin/mim"

echo "Installing MMEngine..."
$MIM_CMD install mmengine

echo "Installing MMCV (2.1.0 for CUDA 12.1)..."
# Force specific version to match torch
$MIM_CMD install "mmcv==2.1.0"

echo "Installing MMDetection (3.1.0 - 3.3.0)..."
$MIM_CMD install "mmdet>=3.1.0,<3.3.0"

echo "Installing MMPose (>=1.1.0)..."
$MIM_CMD install "mmpose>=1.1.0"

# Download RTMO Config and Checkpoint if missing
if [ ! -f "configs/rtmo-l_16xb16-600e_coco-640x640.py" ]; then
    echo "Downloading RTMO config..."
    mkdir -p configs
    $MIM_CMD download mmpose --config rtmo-l_16xb16-600e_coco-640x640 --dest configs
fi

if [ ! -f "model_weights/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth" ]; then
    echo "Moving RTMO checkpoint to model_weights/..."
    mv configs/*.pth model_weights/ 2>/dev/null || true
fi

deactivate


# -------------------------------------------------------
# 4. MoveNet (TensorFlow)
# -------------------------------------------------------
echo ""
echo "-------------------------------------------------"
echo "4. Setting up MoveNet environment (.movenet)"
echo "-------------------------------------------------"
if [ -d ".movenet" ]; then
    echo "Environment .movenet already exists. Skipping creation."
else
    echo "Creating venv .movenet (Python 3.10)..."
    uv venv .movenet --python 3.10
fi

echo "Installing MoveNet dependencies..."
source .movenet/bin/activate
uv pip install tensorflow tensorflow-hub opencv-python tqdm --python .movenet/bin/python --active
deactivate

echo ""
echo "================================================="
echo "  All environments setup successfully! "
echo "================================================="
echo "Usage:"
echo "  .mediapipe/bin/python video_pose_mediapipe.py"
echo "  .yolopose/bin/python video_pose_yolo.py"
echo "  .mmpose/bin/python video_pose_mmpose.py"
echo "  .movenet/bin/python video_pose_movenet.py"

