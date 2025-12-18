#!/bin/bash
set -e

# 1. Sanitize Environment (Fix Cursor AppImage leak)
unset LD_LIBRARY_PATH
echo "Sanitized LD_LIBRARY_PATH..."

# 2. Clean up old venv
echo "Removing old .venv..."
rm -rf .venv

# 3. Create new venv with Python 3.11 (avoids conflict with Cursor's Python 3.10)
echo "Creating venv with Python 3.11..."
uv venv .venv --python 3.11

# 4. Activate venv
source .venv/bin/activate

# 5. Install Base Dependencies (PyTorch, OpenCV, etc.)
echo "Installing PyTorch and base deps..."
uv pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121
uv pip install opencv-python tqdm openmim setuptools tomli platformdirs packaging

# 6. Install MM-Libraries Dependencies manually (skip chumpy for now if possible, install munkres)
echo "Installing helper deps..."
uv pip install munkres xtcocotools json_tricks

# 7. Install MM-Libraries
echo "Installing MM-Libraries..."
# Install engine and cv
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0,<3.3.0"

# Install mmpose without deps to avoid chumpy build failure, since we installed critical deps manually
pip install --no-deps "mmpose>=1.1.0"

# 8. Force Numpy Downgrade (Binary compatibility fix)
echo "Downgrading Numpy to <2.0..."
pip install "numpy<2.0"

echo "========================================"
echo "Environment setup complete!"
echo "Run your script with:"
echo "  unset LD_LIBRARY_PATH && .venv/bin/python -m shot_detector.extract_shots"
echo "========================================"


