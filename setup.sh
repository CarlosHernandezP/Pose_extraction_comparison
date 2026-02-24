#!/bin/bash
set -e  # Exit on error

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# 1. Sanitize Environment (Fix Cursor AppImage leak)
unset LD_LIBRARY_PATH
echo "Sanitized LD_LIBRARY_PATH..."

# 2. Clean up old venv if it exists
if [ -d ".venv" ]; then
    echo "Removing old .venv..."
    rm -rf .venv
fi

# 3. Create virtual environment with Python 3.10
echo "Creating virtual environment with Python 3.10..."
uv venv .venv --python 3.10

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Install Base Dependencies (PyTorch, OpenCV, etc.)
echo "Installing PyTorch and base deps..."
uv pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121

# 6. Install setuptools FIRST (essential for MIM to run)
echo "Installing setuptools (required for MIM)..."
uv pip install setuptools


uv pip install chumpy --no-build-isolation
# 7. Install OpenMIM and other dependencies
echo "Installing OpenMIM and other dependencies..."
uv pip install opencv-python tqdm pandas numpy scikit-learn joblib matplotlib seaborn
uv pip install openmim tomli platformdirs packaging

# 8. Install helper dependencies for MM-Libraries
echo "Installing helper deps..."
uv pip install munkres xtcocotools json_tricks

# 9. Install MM-Libraries (use python -m mim to ensure venv Python is used)
echo "Installing MM-Libraries..."
# Install engine and cv
python -m mim install mmengine
python -m mim install "mmcv==2.1.0"
python -m mim install "mmdet>=3.1.0,<3.3.0"

# Install mmpose without deps to avoid chumpy build failure, since we installed critical deps manually
pip install --no-deps "mmpose>=1.1.0"

# 10. Force Numpy version (Binary compatibility fix)
echo "Ensuring Numpy <2.0..."
pip install "numpy<2.0"

# 11. Download MMPose configs and checkpoints if not present
echo "Checking for MMPose configs and checkpoints..."
if [ ! -f "configs/rtmo-s_8xb32-600e_coco-640x640.py" ]; then
    echo "Downloading RTMO-s config..."
    python -m mim download mmpose --config rtmo-s_8xb32-600e_coco-640x640 --dest configs
fi

if [ ! -f "model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth" ]; then
    echo "Downloading RTMO-s checkpoint..."
    python -m mim download mmpose --config rtmo-s_8xb32-600e_coco-640x640 --dest model_weights
    # Move .pth file from configs to model_weights if needed
    if [ -f "configs/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth" ]; then
        mv configs/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth model_weights/
    fi
fi

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run scripts, use:"
echo "  unset LD_LIBRARY_PATH && .venv/bin/python -m shot_detector.extract_shots"
echo "  unset LD_LIBRARY_PATH && .venv/bin/python -m shot_detector.train_model"
echo "  unset LD_LIBRARY_PATH && .venv/bin/python -m shot_detector.extract_clips_with_ball"
echo ""
echo "Or activate first:"
echo "  source .venv/bin/activate"
echo "  python -m shot_detector.extract_shots"
echo "========================================"

