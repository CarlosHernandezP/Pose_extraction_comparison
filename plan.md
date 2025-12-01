# Pose estimation algorithm comparison


Alright, on this directory I want to create multiple virtual environments and on each one of them I will use a different pose estimation model. 

Main algorithms to try: 

- MediaPipe with GPU
  - **Script:** [video_pose_mediapipe.py](./video_pose_mediapipe.py)
  - **Status:** ✅ GPU Verified 300 frames/second needs to run twice
  - **Implementation Details:** 
    - Switched from the legacy `mp.solutions` API to the new **MediaPipe Tasks API** (`mp.tasks.vision.PoseLandmarker`).
    - Explicitly enabled GPU delegation via `BaseOptions.Delegate.GPU`.
    - Requires downloading the specific `.task` model bundle (e.g., `model_weights/pose_landmarker_full.task`) instead of the default internal model.
- YOLO-Pose
  - **Script:** [video_pose_yolo.py](./video_pose_yolo.py)
  - **Status:** ✅ GPU Verified at 200 frames/second
  - **Implementation Details:** 
    - Uses `ultralytics` package with `model_weights/yolo11n-pose.pt` model.
    - Optimized stream inference.
- MMpose with RTMO
  - **Script:** [video_pose_mmpose.py](./video_pose_mmpose.py)
  - **Status:** 🚧 Setup in progress
  - **Implementation Details:** 
    - Uses `MMPoseInferencer` with `rtmo-l` model.
    - Automatic caching of models via MIM.
  - MoveNet (TensorFlow)
    - **Script:** [video_pose_movenet.py](./video_pose_movenet.py)
    - **Status:** 🚧 Setup in progress
    - **Implementation Details:** 
      - Uses `tensorflow` and `tensorflow_hub` with `movenet/multipose/lightning/1` model.
      - Supports detection of up to 6 people.
      - Input resized to 256x256 (padded).
