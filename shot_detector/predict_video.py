"""
Real-time shot prediction on video.

Processes video frame by frame, tracks bottom-right player, and predicts shots
every 10 frames using a sliding window of 30 frames.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Tuple
from mmpose.apis import MMPoseInferencer
import torch

import os
import sys
from shot_detector.utils import (
    load_fisheye_params, load_perspective_matrix, transform_points, 
    get_foot_position, unwrap_bbox, identify_player
)
from shot_detector.extract_shots import (
    is_pose_valid, filter_stationary_poses, filter_flickering_poses,
    normalize_keypoints_body_relative, COCO_TO_FEATURE_IDX, BODY_KEYPOINT_NAMES
)
from shot_detector.temporal_features import extract_temporal_features

# Configuration
MODEL_CONFIG = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
MODEL_CHECKPOINT = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQUENCE_LENGTH = 30
PREDICTION_STEP = 10  # Predict every 10 frames

# Calibration constants
PARAM_DIR = 'parameters'
FISHEYE_FILE = 'fishcam-fisheye.txt'


def init_mmpose():
    """Initialize MMPose inferencer."""
    if not Path(MODEL_CONFIG).exists():
        raise FileNotFoundError(f"Config not found: {MODEL_CONFIG}")
    if not Path(MODEL_CHECKPOINT).exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CHECKPOINT}")
    
    print(f"Initializing MMPose with {MODEL_CONFIG} on {DEVICE}...")
    return MMPoseInferencer(
        pose2d=MODEL_CONFIG,
        pose2d_weights=MODEL_CHECKPOINT,
        device=DEVICE
    )


def match_player_by_position(poses, prev_bbox, prev_court_pos, K, D, H, exclude_idx=-1):
    """
    Match a player by pixel or court position.
    Returns (match_idx, new_bbox, new_court_pos).
    """
    if not poses:
        return -1, None, None
    
    match_idx = -1
    best_pixel_dist = float('inf')
    best_court_dist = float('inf')
    DISTANCE_THRESHOLD = 150 * 150  # 150px squared
    
    # Try pixel-based matching first
    if prev_bbox is not None:
        cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2
        cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2
        
        for p_idx, p in enumerate(poses):
            if p_idx == exclude_idx:
                continue
            bbox = unwrap_bbox(p['bbox'])
            if len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = (cx - cx_prev)**2 + (cy - cy_prev)**2
            if dist < best_pixel_dist:
                best_pixel_dist = dist
                match_idx = p_idx
    
    # Use pixel match if close enough
    if match_idx != -1 and best_pixel_dist < DISTANCE_THRESHOLD:
        new_bbox = unwrap_bbox(poses[match_idx]['bbox'])
        foot_pos = get_foot_position(new_bbox)
        new_court_pos = None
        if H is not None:
            transformed = transform_points([foot_pos], K, D, H)
            if len(transformed) > 0:
                new_court_pos = transformed[0]
        return match_idx, new_bbox, new_court_pos
    
    # Try court-based recovery
    if prev_court_pos is not None and H is not None:
        match_idx = -1
        COURT_DISTANCE_THRESHOLD = 2.0 * 2.0  # 2 meters
        for p_idx, p in enumerate(poses):
            if p_idx == exclude_idx:
                continue
            bbox = unwrap_bbox(p['bbox'])
            if len(bbox) < 4:
                continue
            foot_pos = get_foot_position(bbox)
            transformed = transform_points([foot_pos], K, D, H)
            if len(transformed) > 0:
                court_pos = transformed[0]
                court_dist = ((court_pos[0] - prev_court_pos[0])**2 + 
                             (court_pos[1] - prev_court_pos[1])**2)
                if court_dist < best_court_dist:
                    best_court_dist = court_dist
                    match_idx = p_idx
        
        if match_idx != -1 and best_court_dist < COURT_DISTANCE_THRESHOLD:
            new_bbox = unwrap_bbox(poses[match_idx]['bbox'])
            foot_pos = get_foot_position(new_bbox)
            transformed = transform_points([foot_pos], K, D, H)
            new_court_pos = transformed[0] if len(transformed) > 0 else None
            return match_idx, new_bbox, new_court_pos
    
    return -1, None, None


def normalize_pose_frame(pose, image_width: int, image_height: int) -> Optional[np.ndarray]:
    """
    Normalize a single pose frame to 27 features.
    
    Parameters
    ----------
    pose : dict
        Pose dict from MMPose with 'keypoints'
    image_width : int
        Image width
    image_height : int
        Image height
        
    Returns
    -------
    np.ndarray or None
        Array of 27 normalized features, or None if invalid
    """
    if 'keypoints' not in pose:
        return None
    
    try:
        kpts = np.array(pose['keypoints'])
        
        # Handle different keypoint formats
        if len(kpts.shape) == 1:
            if len(kpts) >= 34:
                kpts = kpts.reshape(-1, 2)[:17]
            else:
                return None
        elif len(kpts.shape) == 2:
            if kpts.shape[0] < 17:
                padded = np.zeros((17, 2))
                padded[:kpts.shape[0], :2] = kpts[:, :2]
                kpts = padded
        else:
            return None
        
        # Extract 12 body keypoints in correct order
        body_kpts_12 = np.zeros((12, 2))
        for coco_idx, feat_idx in COCO_TO_FEATURE_IDX.items():
            if coco_idx < len(kpts):
                kp = kpts[coco_idx]
                if len(kp) >= 2:
                    body_kpts_12[feat_idx] = kp[:2]
        
        # Normalize
        features = normalize_keypoints_body_relative(body_kpts_12, image_width, image_height)
        return features
        
    except Exception as e:
        return None


def draw_prediction_bars(frame: np.ndarray, predictions: Dict[str, float], 
                        class_names: list, bar_width: int = 200, bar_height: int = 30):
    """
    Draw horizontal probability bars in top right corner.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame to draw on
    predictions : dict
        Dictionary mapping class names to probabilities
    class_names : list
        List of class names in order
    bar_width : int
        Width of each bar
    bar_height : int
        Height of each bar
    """
    height, width = frame.shape[:2]
    
    # Position in top right
    bar_box_x = width - 280
    bar_box_y = 20
    bar_box_width = 260
    bar_spacing = 40
    start_y = bar_box_y + 20
    
    # Background box (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_box_x, bar_box_y), 
                 (bar_box_x + bar_box_width, bar_box_y + len(class_names) * bar_spacing + 20), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Colors for each class
    colors = {
        'forehand': (255, 0, 0),    # Blue (BGR)
        'backhand': (0, 255, 0),    # Green (BGR)
        'serve': (0, 0, 255),        # Red (BGR)
        'idle': (128, 128, 128),     # Gray (BGR)
    }
    
    # Draw bars for each class
    for i, class_name in enumerate(class_names):
        y_pos = start_y + i * bar_spacing
        
        # Get probability
        prob = predictions.get(class_name, 0.0)
        
        # Class label
        label = f"{class_name}: {prob:.2f}"
        cv2.putText(frame, label, (bar_box_x + 5, y_pos - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bar background (gray)
        bar_x = bar_box_x + 5
        cv2.rectangle(frame, (bar_x, y_pos), 
                     (bar_x + bar_width, y_pos + bar_height), 
                     (100, 100, 100), -1)
        
        # Bar fill (colored)
        fill_width = int(bar_width * prob)
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(frame, (bar_x, y_pos), 
                     (bar_x + fill_width, y_pos + bar_height), 
                     color, -1)
        
        # Bar border
        cv2.rectangle(frame, (bar_x, y_pos), 
                     (bar_x + bar_width, y_pos + bar_height), 
                     (255, 255, 255), 1)


def predict_video(
    video_path: str,
    model_path: str,
    label_encoder_path: Optional[str] = None,
    output_path: Optional[str] = None,
    calibration_video_name: Optional[str] = None
):
    """
    Process video and predict shots in real-time.
    
    Parameters
    ----------
    video_path : str
        Path to input video
    model_path : str
        Path to trained Random Forest model
    label_encoder_path : str, optional
        Path to label encoder (auto-detected if None)
    output_path : str, optional
        Path to save output video (if None, saves next to input)
    calibration_video_name : str, optional
        Video name to determine calibration files (e.g., 'BO-0001' for BO01 files)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    rf_model = joblib.load(model_path)
    
    if label_encoder_path is None:
        label_encoder_path = Path(model_path).parent / "label_encoder_rf_cv.pkl"
    if not Path(label_encoder_path).exists():
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
    
    label_encoder = joblib.load(label_encoder_path)
    class_names = label_encoder.classes_
    print(f"Model loaded. Classes: {class_names}")
    
    # Load calibration
    K, D = load_fisheye_params(os.path.join(PARAM_DIR, FISHEYE_FILE))
    
    # Determine perspective matrix file
    H = None
    if calibration_video_name:
        # Try to find matching perspective file
        if 'BO' in calibration_video_name:
            persp_file = os.path.join(PARAM_DIR, 'BO01-perspective.txt')
            if not Path(persp_file).exists():
                persp_file = os.path.join(PARAM_DIR, 'BO02-perspective.txt')
        elif 'LU' in calibration_video_name:
            persp_file = os.path.join(PARAM_DIR, 'LU01-perspective.txt')
            if not Path(persp_file).exists():
                persp_file = os.path.join(PARAM_DIR, 'LU02-perspective.txt')
        else:
            persp_file = None
        
        if persp_file and Path(persp_file).exists():
            H = load_perspective_matrix(persp_file)
            print(f"Loaded perspective matrix from {persp_file}")
    
    # Initialize MMPose
    inferencer = init_mmpose()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Setup output video
    if output_path is None:
        output_path = Path(video_path).parent / f"{Path(video_path).stem}_predictions.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Pose buffer for prediction (30 frames)
    pose_buffer = []  # List of normalized_features or None
    tracked_player_idx = -1
    tracked_bbox = None
    tracked_court_pos = None
    last_valid_pose_features = None
    
    # Small buffer for pose filtering (last 5 frames for basic filtering)
    recent_poses_buffer = []  # List of lists of poses per frame
    
    # Current predictions (updated every 10 frames)
    current_predictions = {cls: 0.0 for cls in class_names}
    
    frame_num = 0
    print("\nProcessing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose estimation
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        predictions = result['predictions'][0]
        
        # Filter valid poses using same logic as extract_shots.py
        valid_poses = []
        for pose in predictions:
            if is_pose_valid(pose, K, D, H, img_height=height, img_width=width):
                valid_poses.append(pose)
        
        # Add to recent buffer for filtering (keep last 5 frames)
        recent_poses_buffer.append(valid_poses)
        if len(recent_poses_buffer) > 5:
            recent_poses_buffer.pop(0)
        
        # Apply basic filtering if we have enough frames
        poses_to_track = valid_poses
        if len(recent_poses_buffer) >= 5:
            # Apply stationary filtering (simplified - only on recent buffer)
            filtered_poses_per_frame = filter_stationary_poses(
                recent_poses_buffer, 
                movement_threshold=20.0, 
                min_frames=3,  # Lower threshold for real-time
                img_height=height, 
                bottom_margin=0.05,
                filter_bottom_stationary_only=True
            )
            # Get poses from last frame
            if filtered_poses_per_frame and len(filtered_poses_per_frame) > 0:
                poses_to_track = filtered_poses_per_frame[-1]
            else:
                poses_to_track = valid_poses
        
        # Identify bottom-right player (first frame or if lost)
        if tracked_player_idx == -1 or tracked_player_idx >= len(poses_to_track):
            # Try to identify bottom-right player
            if poses_to_track:
                # Find player with highest Y (bottom) and high X (right)
                # Bottom-right = high Y (bottom of image) AND high X (right of image)
                best_score = float('-inf')
                for i, pose in enumerate(poses_to_track):
                    bbox = unwrap_bbox(pose['bbox'])
                    if len(bbox) >= 4:
                        foot_pos = get_foot_position(bbox)
                        foot_x, foot_y = foot_pos
                        # Score: prefer bottom-right corner
                        # Weight Y more (bottom) and X (right)
                        # Normalize: Y/height gives 0-1, X/width gives 0-1
                        score = (foot_y / height) * 0.7 + (foot_x / width) * 0.3
                        if score > best_score:
                            best_score = score
                            tracked_player_idx = i
        
        # Track player across frames
        if tracked_player_idx != -1 and tracked_player_idx < len(poses_to_track):
            # Update tracking
            tracked_pose = poses_to_track[tracked_player_idx]
            tracked_bbox = unwrap_bbox(tracked_pose['bbox'])
            foot_pos = get_foot_position(tracked_bbox)
            if H is not None:
                transformed = transform_points([foot_pos], K, D, H)
                if len(transformed) > 0:
                    tracked_court_pos = transformed[0]
        else:
            # Try to recover by position matching
            if tracked_bbox is not None:
                match_idx, new_bbox, new_court_pos = match_player_by_position(
                    poses_to_track, tracked_bbox, tracked_court_pos, K, D, H
                )
                if match_idx != -1:
                    tracked_player_idx = match_idx
                    tracked_bbox = new_bbox
                    tracked_court_pos = new_court_pos
                else:
                    tracked_player_idx = -1
            else:
                tracked_player_idx = -1
        
        # Get normalized features for tracked player
        pose_features = None
        if tracked_player_idx != -1 and tracked_player_idx < len(poses_to_track):
            pose_features = normalize_pose_frame(
                poses_to_track[tracked_player_idx], width, height
            )
            if pose_features is not None:
                last_valid_pose_features = pose_features
        
        # Use last valid pose if current is None (forward fill)
        if pose_features is None and last_valid_pose_features is not None:
            pose_features = last_valid_pose_features.copy()
        
        # Add to buffer
        pose_buffer.append(pose_features)
        
        # Keep only last 30 frames
        if len(pose_buffer) > SEQUENCE_LENGTH:
            pose_buffer.pop(0)
        
        # Predict every PREDICTION_STEP frames (if we have 30 frames)
        if len(pose_buffer) >= SEQUENCE_LENGTH and (frame_num + 1) % PREDICTION_STEP == 0:
            # Check if we have enough valid poses
            valid_count = sum(1 for p in pose_buffer[-SEQUENCE_LENGTH:] if p is not None)
            if valid_count >= SEQUENCE_LENGTH * 0.5:  # At least 50% valid
                # Get last 30 frames
                recent_buffer = pose_buffer[-SEQUENCE_LENGTH:]
                
                # Fill None with last valid or zeros
                sequence = []
                last_valid = None
                for p in recent_buffer:
                    if p is not None:
                        sequence.append(p)
                        last_valid = p
                    elif last_valid is not None:
                        sequence.append(last_valid)
                    else:
                        # Fallback to zeros if no valid pose yet
                        sequence.append(np.zeros(27))
                
                # Convert to array and reshape for temporal features
                sequence_array = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 27)
                
                # Extract temporal features
                X = extract_temporal_features(sequence_array)
                
                # Predict
                probabilities = rf_model.predict_proba(X)[0]
                
                # Update current predictions
                for i, cls in enumerate(class_names):
                    current_predictions[cls] = float(probabilities[i])
        
        # Draw visualization
        vis_frame = frame.copy()
        
        # Draw bounding box of tracked player
        if tracked_player_idx != -1 and tracked_player_idx < len(poses_to_track):
            bbox = unwrap_bbox(poses_to_track[tracked_player_idx]['bbox'])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, "TRACKED", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw prediction bars
        draw_prediction_bars(vis_frame, current_predictions, class_names)
        
        # Draw frame info
        info_text = f"Frame: {frame_num} | Buffer: {len(pose_buffer)}/30"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(vis_frame)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames")
    
    cap.release()
    out.release()
    print(f"\nOutput video saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict shots on video in real-time")
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument(
        '--model-path',
        type=str,
        default='model_weights/random_forest_model_cv.pkl',
        help='Path to trained model (default: model_weights/random_forest_model_cv.pkl)'
    )
    parser.add_argument(
        '--label-encoder-path',
        type=str,
        default=None,
        help='Path to label encoder (auto-detected if None)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Path to save output video (default: {input}_predictions.mp4)'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default=None,
        help='Video name for calibration (e.g., BO-0001, LU-0002)'
    )
    
    args = parser.parse_args()
    
    # Extract calibration name from video path if not provided
    calibration_name = args.calibration
    if calibration_name is None:
        video_name = Path(args.video_path).stem
        # Try to extract camera ID (e.g., BO-0001, LU-0002)
        import re
        match = re.search(r'(BO|LU)-\d+', video_name)
        if match:
            calibration_name = match.group(0)
    
    predict_video(
        video_path=args.video_path,
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        output_path=args.output_path,
        calibration_video_name=calibration_name
    )


if __name__ == "__main__":
    import os
    import sys
    main()
