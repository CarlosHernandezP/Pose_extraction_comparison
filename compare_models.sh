#!/bin/bash

# Configuration
VIDEO_PATH="$(pwd)/data/14-10-BO-0001_short.mp4"
LOG_FILE="comparison_results.txt"
RESULTS_DIR="results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Clear previous log file
echo "Pose Estimation Comparison - $(date)" > "$LOG_FILE"
echo "Video: $VIDEO_PATH" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

echo "Starting comparison..."
echo "Video path: $VIDEO_PATH"

# Function to run a test
run_test() {
    local venv_name=$1
    local script_name=$2
    local model_name=$3
    local model_slug=$4
    
    echo "========================================"
    echo "Running $model_name..."
    echo "========================================"
    
    local output_dir="$RESULTS_DIR/$model_slug"
    mkdir -p "$output_dir"
    local output_file="$output_dir/output.mp4"
    
    # Run in a subshell to isolate environment activation
    (
        source "$venv_name/bin/activate"
        
        # Check if python is available in venv
        if ! command -v python &> /dev/null; then
             echo "Error: Python not found in $venv_name"
             exit 1
        fi
        
        # Run the script and append output to log
        # We also tee it to stdout so user sees progress
        # Passing output_file as the second argument enables saving
        python "$script_name" "$VIDEO_PATH" "$output_file" | tee -a "$LOG_FILE"
    )
    
    echo "" >> "$LOG_FILE"
}

# 1. MediaPipe
if [ -d ".mediapipe" ]; then
    run_test ".mediapipe" "video_pose_mediapipe.py" "MediaPipe (GPU)" "mediapipe"
else
    echo "Skipping MediaPipe: .mediapipe venv not found"
fi

# 2. YOLO-Pose
if [ -d ".yolopose" ]; then
    run_test ".yolopose" "video_pose_yolo.py" "YOLO-Pose" "yolo"
else
    echo "Skipping YOLO-Pose: .yolopose venv not found"
fi

# 3. MMPose
if [ -d ".mmpose" ]; then
    run_test ".mmpose" "video_pose_mmpose.py" "MMPose (RTMO)" "mmpose"
else
    echo "Skipping MMPose: .mmpose venv not found"
fi

# 4. MoveNet
if [ -d ".movenet" ]; then
    run_test ".movenet" "video_pose_movenet.py" "MoveNet" "movenet"
else
    echo "Skipping MoveNet: .movenet venv not found"
fi

echo "========================================"
echo "Comparison finished!"
echo "Results saved to $LOG_FILE"
