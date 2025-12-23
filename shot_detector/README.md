# Shot Detector

This module implements a pipeline for extracting padel shot pose data from videos and training a Random Forest classifier for shot detection.

## Structure

- `extract_shots.py`: Main script. Reads CSVs, extracts video clips, runs MMPose, and saves normalized pose data.
- `train_model.py`: Script for training the Random Forest classifier.
- `temporal_features.py`: Temporal feature extraction (replicated from LookAtMeProtoApp).
- `shot_mapper.py`: Flexible shot type mapping system.
- `data_augmentation.py`: Data augmentation for increasing dataset size.
- `utils.py`: Helper functions for CSV parsing and player identification.
- `data/`: Output directory for generated CSVs and video clips.

## Data Creation

### 1. Extract Pose Data from Videos

Run the extraction script to generate normalized pose sequences:

```bash
python -m shot_detector.extract_shots
```

*Note: Run from the project root (`/home/carlos/pose_estimators/`) so that config paths resolve correctly.*

**What it does:**
- Reads shot annotation CSVs from `/home/daniele/shots_csvs/`
- Reads videos from `/home/daniele/videos/`
- For each shot, extracts 30 frames (15 before + center + 14 after)
- Runs MMPose to detect poses
- Normalizes poses using body-relative coordinates (27 features per frame)
- Saves normalized pose data to `shot_detector/data/`

**Output format:**
- `{video}_{frame}_{shot_type}_{player}_pose.csv`: Normalized pose features for active player (27 columns)
- `{video}_{frame}_idle_{player}_pose.csv`: Normalized pose features for idle player (27 columns)
- `{video}_{frame}_{shot_type}_{player}.mp4`: Verification clip with overlaid poses (Green = Active, Red = Idle)

**Pose CSV format:**
- `frame_num`: Frame index (0-29)
- 24 body-relative features: `{keypoint}_x_body_rel`, `{keypoint}_y_body_rel` for 12 keypoints
- 3 absolute position features: `hip_y_abs`, `hip_x_abs`, `shoulder_center_y_abs`

## Training

### 1. Discover Shot Types

First, check what shot types are in your data:

```bash
python -m shot_detector.train_model --discover-shots --data-dir shot_detector/data
```

This will show all unique shot types and indicate which are mapped to classes.

### 2. Train the Model

Train the Random Forest classifier:

```bash
python -m shot_detector.train_model --data-dir shot_detector/data --output-dir model_weights
```

**Options:**
- `--data-dir`: Directory containing pose CSV files (default: `shot_detector/data`)
- `--output-dir`: Directory to save model (default: `model_weights`)
- `--no-augmentation`: Disable data augmentation
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth (default: None)
- `--cv-folds`: Cross-validation folds (default: 5)

**What it does:**
- Loads all pose CSV files from data directory
- Extracts shot types from filenames (including 'idle' for idle player poses)
- Maps shot types to 4 classes: `forehand`, `backhand`, `serve`, `idle`
- Applies data augmentation (mirroring, temporal warping, noise)
- Extracts temporal features from 30-frame sequences
- Trains Random Forest classifier
- Performs cross-validation evaluation
- Saves model to `model_weights/random_forest_model_cv.pkl`
- Saves label encoder to `model_weights/label_encoder_rf_cv.pkl`

**Model compatibility:**
The trained model is compatible with `shot_predictor.py` from LookAtMeProtoApp and can be used directly for inference.
