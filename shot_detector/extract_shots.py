import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import re
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import parse_shot_csv, get_video_path, identify_player, get_idle_player
from shot_detector.utils import load_fisheye_params, load_perspective_matrix, transform_points, get_foot_position

# Configuration
SHOTS_CSV_DIRS = [
    'shot_csvs/shots_csvs/',
    '/home/daniele/shots_csvs/'
]
VIDEOS_DIRS = [
    'videos/',
    '/home/daniele/videos/'
]
OUTPUT_DIR = 'shot_detector/data/'
MODEL_CONFIG = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
MODEL_CHECKPOINT = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Calibration constants
PARAM_DIR = 'parameters'
FISHEYE_FILE = 'fishcam-fisheye.txt'

# Debug mode: Set to True to see all detected poses with indices
DEBUG_MODE = True

# Tracking thresholds
DISTANCE_THRESHOLD = 150 * 150  # 150px squared (pixel space)
COURT_DISTANCE_THRESHOLD = 2.0 * 2.0  # 2 meters in court space

def init_mmpose():
    if not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Config not found: {MODEL_CONFIG}")
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CHECKPOINT}")
    
    print(f"Initializing MMPose with {MODEL_CONFIG} on {DEVICE}...")
    print(f"Detection score threshold: 0.05 (lowered from 0.1 in config)")
    return MMPoseInferencer(
        pose2d=MODEL_CONFIG,
        pose2d_weights=MODEL_CHECKPOINT,
        device=DEVICE
    )

def match_player_by_position(poses, prev_bbox, prev_court_pos, K, D, H, exclude_idx=-1):
    """Match a player by pixel or court position. Returns (match_idx, new_bbox, new_court_pos)."""
    if not poses:
        return -1, None, None
    
    match_idx = -1
    best_pixel_dist = float('inf')
    best_court_dist = float('inf')
    
    # Try pixel-based matching first
    if prev_bbox is not None:
        cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2
        cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2
        
        for p_idx, p in enumerate(poses):
            if p_idx == exclude_idx: continue
            bbox = unwrap_bbox(p['bbox'])
            if len(bbox) < 4: continue
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
        for p_idx, p in enumerate(poses):
            if p_idx == exclude_idx: continue
            bbox = unwrap_bbox(p['bbox'])
            if len(bbox) < 4: continue
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
            new_court_pos = None
            if H is not None:
                transformed = transform_points([foot_pos], K, D, H)
                if len(transformed) > 0:
                    new_court_pos = transformed[0]
            return match_idx, new_bbox, new_court_pos
    
    # No match found
    return -1, None, prev_court_pos  # Keep court position for next frame

def get_calibration(video_name):
    """
    Infers camera parameters based on video name.
    """
    # Common fisheye params
    fisheye_path = os.path.join(PARAM_DIR, FISHEYE_FILE)
    K, D = load_fisheye_params(fisheye_path)
    
    # Perspective matrix based on camera name
    camera_name = None
    
    # Check standard names BO01, BO02...
    for cam in ['BO01', 'BO02', 'LU01', 'LU02']:
        if cam in video_name: 
            camera_name = cam
            break
            
    # Try with hyphenated version (e.g. BO-01, BO-0001)
    if not camera_name:
        for prefix in ['BO', 'LU']:
            # Search for pattern PREFIX-NUMBER
            match = re.search(f"{prefix}-(\d+)", video_name)
            if match:
                num = int(match.group(1))
                # Normalize to 2 digits: BO01
                camera_name = f"{prefix}{num:02d}"
                break

    H = None
    if camera_name:
        perspective_path = os.path.join(PARAM_DIR, f"{camera_name}-perspective.txt")
        H = load_perspective_matrix(perspective_path)
    else:
        # Fallback or specific logic for other names?
        pass

    return K, D, H

def is_pose_valid(pose, K, D, H, img_height=None):
    """
    Checks if a pose is within the imaginary field (0-10, 0-20).
    Also filters background players (y < 30% of image height).
    """
    bbox = unwrap_bbox(pose['bbox'])
    if len(bbox) < 4:
        return False
        
    # 1. Pixel-based filtering (y > 30% of height)
    if img_height is not None:
        foot_y = get_foot_position(bbox)[1]
        if foot_y < 0.25 * img_height:
             return False # Too far up (background players)

    if H is None:
        return True # No calibration, assume valid or handle differently
        
    foot_pos = get_foot_position(bbox)
    
    # Transform
    # transform_points expects list of points or (N,2)
    # It returns (N,2)
    transformed = transform_points([foot_pos], K, D, H)
    if len(transformed) == 0:
        return False
        
    tx, ty = transformed[0]
    
    # Filter -0.3 to 10.3 x (0.3m margin), -1.3 to 21.3 y (1.3m margin)
    if -0.3 <= tx <= 10.3 and -1.3 <= ty <= 21.3:
        return True
        
    return False

def extract_clip_and_pose(video_path, start_frame, duration, inferencer, K=None, D=None, H=None, return_all_poses=False):
    """
    Extracts frames, runs pose estimation, and returns data.
    Filters poses based on calibration if provided.
    
    Args:
        return_all_poses: If True, returns all poses (filtered and unfiltered) for debugging
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    poses_per_frame = []
    all_poses_per_frame = [] if return_all_poses else None
    
    for _ in range(duration):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Run inference on single frame
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        
        predictions = result['predictions'][0]
        
        frame_poses = []
        frame_all_poses = []
        
        if predictions:
            for instance in predictions:
                if return_all_poses:
                    # Store all poses with validity flag
                    is_valid = is_pose_valid(instance, K, D, H, img_height=img_height)
                    frame_all_poses.append((instance, is_valid))
                
                # Filter here
                if is_pose_valid(instance, K, D, H, img_height=img_height):
                    frame_poses.append(instance)
                
        poses_per_frame.append(frame_poses)
        if return_all_poses:
            all_poses_per_frame.append(frame_all_poses)

    cap.release()
    if return_all_poses:
        return frames, poses_per_frame, all_poses_per_frame
    return frames, poses_per_frame

def unwrap_bbox(bbox):
    """
    Helper to unwrap nested bbox structure ([x1,y1,x2,y2],) -> [x1,y1,x2,y2]
    """
    if len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
        return bbox[0]
    return bbox

def draw_overlay(frame, poses, active_idx, idle_idx, debug=False, all_poses_info=None, 
                is_forward_filled=False, forward_filled_pose=None):
    """
    Draws bounding boxes and labels on the frame.
    
    Args:
        debug: If True, shows all poses with indices and debug info
        all_poses_info: List of (pose, is_valid) tuples for all detected poses (including filtered ones)
        is_forward_filled: If True, active player is forward-filled (show in blue)
        forward_filled_pose: The pose to draw when forward-filling (if None, uses last pose in poses)
    """
    img = frame.copy()
    
    # Only draw poses that are being tracked
    tracked_indices = set()
    if active_idx != -1:
        tracked_indices.add(active_idx)
    if idle_idx != -1:
        tracked_indices.add(idle_idx)
    
    # If all_poses_info is provided, draw ALL poses (filtered and unfiltered)
    if all_poses_info is not None:
        fill_status = "FILLED" if is_forward_filled else ""
        debug_text = f"All detected: {len(all_poses_info)} | Valid: {len(poses)} | Active: {active_idx} | Idle: {idle_idx} {fill_status}"
        cv2.putText(img, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw all poses
        valid_count = 0
        for i, (pose, is_valid) in enumerate(all_poses_info):
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4: continue
            
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Color coding:
            # - Green: Valid AND tracked as Active
            # - Red: Valid AND tracked as Idle  
            # - Yellow: Valid but not tracked
            # - Magenta: Invalid (filtered out)
            
            if is_valid:
                valid_count += 1
                # Check if this pose is in the filtered list and tracked
                # Compare by bbox to find matching pose
                pose_in_filtered = False
                tracked_as = None
                this_bbox = unwrap_bbox(pose['bbox'])
                for j, filtered_pose in enumerate(poses):
                    filtered_bbox = unwrap_bbox(filtered_pose['bbox'])
                    # Compare bboxes (allow small tolerance for floating point)
                    if len(this_bbox) >= 4 and len(filtered_bbox) >= 4:
                        if (abs(this_bbox[0] - filtered_bbox[0]) < 1 and
                            abs(this_bbox[1] - filtered_bbox[1]) < 1 and
                            abs(this_bbox[2] - filtered_bbox[2]) < 1 and
                            abs(this_bbox[3] - filtered_bbox[3]) < 1):
                            pose_in_filtered = True
                            if j == active_idx:
                                tracked_as = 'active'
                            elif j == idle_idx:
                                tracked_as = 'idle'
                            break
                
                if tracked_as == 'active':
                    if is_forward_filled:
                        color = (255, 0, 0)  # Blue - forward-filled
                        label = f"P{i} ACTIVE (FILLED)"
                    else:
                        color = (0, 255, 0)  # Green
                        label = f"P{i} ACTIVE"
                elif tracked_as == 'idle':
                    color = (0, 0, 255)  # Red
                    label = f"P{i} IDLE"
                elif pose_in_filtered:
                    color = (0, 255, 255)  # Yellow - valid but not tracked
                    label = f"P{i} VALID"
                else:
                    color = (0, 255, 255)  # Yellow - should not happen
                    label = f"P{i} VALID?"
            else:
                color = (255, 0, 255)  # Magenta - filtered out
                label = f"P{i} FILTERED"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw keypoints
            if 'keypoints' in pose:
                kpts = pose['keypoints']
                for kp in kpts:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(img, (kx, ky), 3, color, -1)
        
        cv2.putText(img, f"Valid: {valid_count}/{len(all_poses_info)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw forward-filled pose if needed (draw on top of everything)
        if is_forward_filled and forward_filled_pose is not None:
            bbox = unwrap_bbox(forward_filled_pose['bbox'])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                color = (255, 0, 0)  # Blue (BGR format)
                label = "ACTIVE (FILLED)"
                # Draw with thicker line to make it more visible
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                
                # Draw keypoints
                if 'keypoints' in forward_filled_pose:
                    kpts = forward_filled_pose['keypoints']
                    for kp in kpts:
                        kx, ky = int(kp[0]), int(kp[1])
                        if kx > 0 and ky > 0:
                            cv2.circle(img, (kx, ky), 4, color, -1)
            else:
                print(f"WARNING: Forward-filled pose has invalid bbox: {bbox}")
        
        return img
    
    # Original behavior (only filtered poses)
    if debug:
        debug_text = f"Detected: {len(poses)} poses | Active: {active_idx} | Idle: {idle_idx}"
        cv2.putText(img, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(poses) > 0:
            all_indices = ", ".join([str(i) for i in range(len(poses))])
            cv2.putText(img, f"Indices: [{all_indices}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for i, pose in enumerate(poses):
        bbox = unwrap_bbox(pose['bbox'])
        if len(bbox) < 4: continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        if debug:
            color = (128, 128, 128) # Gray for untracked
            label = f"P{i}"
            
            if i == active_idx:
                if is_forward_filled:
                    color = (255, 0, 0)  # Blue - forward-filled
                    label = f"P{i} ACTIVE (FILLED)"
                else:
                    color = (0, 255, 0)  # Green for Active
                    label = f"P{i} ACTIVE"
            elif i == idle_idx:
                color = (0, 0, 255) # Red for Idle
                label = f"P{i} IDLE"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            if i not in tracked_indices:
                continue
                
            color = (255, 255, 255)
            label = ""
            
            if i == active_idx:
                color = (0, 255, 0)
                label = "ACTIVE"
            elif i == idle_idx:
                color = (0, 0, 255)
                label = "IDLE"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if label:
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        if 'keypoints' in pose:
            kpts = pose['keypoints']
            for kp in kpts:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0:
                    cv2.circle(img, (kx, ky), 3, color, -1)
    
    # Draw forward-filled pose if needed (draw on top of everything)
    if is_forward_filled and forward_filled_pose is not None:
        bbox = unwrap_bbox(forward_filled_pose['bbox'])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            color = (255, 0, 0)  # Blue (BGR format)
            label = "ACTIVE (FILLED)"
            # Draw with thicker line to make it more visible
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            
            # Draw keypoints
            if 'keypoints' in forward_filled_pose:
                kpts = forward_filled_pose['keypoints']
                for kp in kpts:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(img, (kx, ky), 4, color, -1)
        else:
            print(f"WARNING: Forward-filled pose has invalid bbox: {bbox}")

    return img

def save_pose_csv(poses_sequence, output_path):
    """
    Saves the pose sequence to a CSV.
    Format: frame_idx, kpt1_x, kpt1_y, kpt1_score, ...
    """
    data = []
    for frame_idx, pose in enumerate(poses_sequence):
        row = {'frame_idx': frame_idx}
        if pose:
            kpts = pose['keypoints'] # usually list of [x, y] or [x, y, score]
            # keypoint_scores might be separate
            kp_scores = pose.get('keypoint_scores', [1.0]*len(kpts))
            
            for i, (kp, score) in enumerate(zip(kpts, kp_scores)):
                row[f'kpt{i}_x'] = kp[0]
                row[f'kpt{i}_y'] = kp[1]
                row[f'kpt{i}_score'] = score
        else:
            # Handle missing pose? Fill 0?
            pass
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def main():
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize MMPose
    inferencer = init_mmpose()
    
    # List CSVs from all directories
    csv_files_map = {} # filename -> full_path
    
    for d in SHOTS_CSV_DIRS:
        if not os.path.exists(d): 
            continue
        for f in os.listdir(d):
            if f.endswith('.csv'):
                csv_files_map[f] = os.path.join(d, f)
    
    if not csv_files_map:
        print("No CSV files found in directories:", SHOTS_CSV_DIRS)
        return

    csv_files = sorted(list(csv_files_map.keys()))
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        csv_path = csv_files_map[csv_file]
        
        df = parse_shot_csv(csv_path)
        if df.empty:
            continue
            
        video_path = get_video_path(csv_path, VIDEOS_DIRS)
        if not video_path:
            tqdm.write(f"Video not found for {csv_file}")
            continue
            
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Get calibration for this video
        K, D, H = get_calibration(video_name)
        if H is None:
            tqdm.write(f"Warning: No perspective matrix found for {video_name}, filtering might be ineffective.")
        
        # Group by row to process each shot
        # Convert to list to use tqdm
        rows = list(df.iterrows())
        for idx, row in tqdm(rows, desc=f"Shots in {csv_file}", leave=False):
            shot_type = row['Shot']
            center_frame = int(row['FrameId'])
            player_label = row['Player']
            
            # Frame range: 15 before, center, 14 after = 30 frames
            # Start = center - 15
            start_frame = max(0, center_frame - 15)
            duration = 30
            
            # Extract data with filtering
            if DEBUG_MODE:
                frames, poses_per_frame, all_poses_per_frame = extract_clip_and_pose(
                    video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=True
                )
            else:
                frames, poses_per_frame = extract_clip_and_pose(
                    video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=False
                )
                all_poses_per_frame = None
            
            if not frames or not poses_per_frame:
                tqdm.write(f"    Failed to extract frames for shot {shot_type} at {center_frame}")
                continue
                
            # Identify players in the CENTER frame (approx index 15)
            center_idx_in_clip = min(15, len(frames) - 1)
            center_poses = poses_per_frame[center_idx_in_clip]
            
            if DEBUG_MODE:
                tqdm.write(f"    Center frame: {len(center_poses)} valid poses, looking for '{player_label}'")
            
            active_idx_initial = identify_player(center_poses, player_label, K, D, H)
            
            if DEBUG_MODE and active_idx_initial != -1:
                # Show transformed position of selected player
                selected_pose = center_poses[active_idx_initial]
                bbox = unwrap_bbox(selected_pose['bbox'])
                foot_pos = get_foot_position(bbox)
                if H is not None:
                    transformed = transform_points([foot_pos], K, D, H)
                    if len(transformed) > 0:
                        tx, ty = transformed[0]
                        tqdm.write(f"    Selected player {active_idx_initial} at court position ({tx:.2f}, {ty:.2f})")
            
            if active_idx_initial == -1:
                tqdm.write(f"    Could not find player '{player_label}' in center frame for shot {shot_type} at {center_frame}")
                continue
                
            idle_idx_initial = get_idle_player(center_poses, active_idx_initial)
            
            tracked_active_indices = {center_idx_in_clip: active_idx_initial}
            tracked_idle_indices = {center_idx_in_clip: idle_idx_initial}
            
            # Prepare init state
            init_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            init_active_foot = get_foot_position(init_active_bbox)
            init_active_court_pos = None
            if H is not None:
                transformed = transform_points([init_active_foot], K, D, H)
                if len(transformed) > 0:
                    init_active_court_pos = transformed[0]
            
            init_idle_bbox = None
            init_idle_court_pos = None
            if idle_idx_initial != -1:
                init_idle_bbox = unwrap_bbox(center_poses[idle_idx_initial]['bbox'])
                init_idle_foot = get_foot_position(init_idle_bbox)
                if H is not None:
                    transformed = transform_points([init_idle_foot], K, D, H)
                    if len(transformed) > 0:
                        init_idle_court_pos = transformed[0]
            
            # --- Forward Pass ---
            curr_active_bbox = init_active_bbox
            curr_active_court_pos = init_active_court_pos
            curr_idle_bbox = init_idle_bbox
            curr_idle_court_pos = init_idle_court_pos
            
            for i in range(center_idx_in_clip + 1, len(frames)):
                poses = poses_per_frame[i]
                active_match_idx = -1
                idle_match_idx = -1
                
                # Track Active: by position first, only re-identify by label if completely lost
                if poses:
                    # Try position-based tracking first
                    active_match_idx, curr_active_bbox, curr_active_court_pos = match_player_by_position(
                        poses, curr_active_bbox, curr_active_court_pos, K, D, H, exclude_idx=-1
                    )
                    
                    # Only re-identify by label if completely lost AND we had no previous idle tracking
                    # (prevents idle from becoming active when both were present)
                    if active_match_idx == -1 and curr_active_bbox is None and curr_active_court_pos is None:
                        # Check if we had idle tracking - if so, don't re-identify (idle can't become active)
                        had_idle_tracking = (curr_idle_bbox is not None or curr_idle_court_pos is not None)
                        if not had_idle_tracking:
                            label_match_idx = identify_player(poses, player_label, K, D, H)
                            if label_match_idx != -1:
                                active_match_idx = label_match_idx
                                curr_active_bbox = unwrap_bbox(poses[label_match_idx]['bbox'])
                                foot_pos = get_foot_position(curr_active_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_active_court_pos = transformed[0]
                
                tracked_active_indices[i] = active_match_idx
                
                # Track Idle: by position, excluding active
                if poses:
                    idle_match_idx, curr_idle_bbox, curr_idle_court_pos = match_player_by_position(
                        poses, curr_idle_bbox, curr_idle_court_pos, K, D, H, exclude_idx=active_match_idx
                    )
                    
                    # If we have active but no idle, and there are multiple poses, assign the other one as idle
                    if active_match_idx != -1 and idle_match_idx == -1 and len(poses) > 1:
                        for p_idx in range(len(poses)):
                            if p_idx != active_match_idx:
                                idle_match_idx = p_idx
                                curr_idle_bbox = unwrap_bbox(poses[p_idx]['bbox'])
                                foot_pos = get_foot_position(curr_idle_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_idle_court_pos = transformed[0]
                                break
                
                tracked_idle_indices[i] = idle_match_idx

            # --- Backward Pass ---
            curr_active_bbox = init_active_bbox
            curr_active_court_pos = init_active_court_pos
            curr_idle_bbox = init_idle_bbox
            curr_idle_court_pos = init_idle_court_pos
            
            for i in range(center_idx_in_clip - 1, -1, -1):
                poses = poses_per_frame[i]
                active_match_idx = -1
                idle_match_idx = -1
                
                # Track Active: by position first, only re-identify by label if completely lost
                if poses:
                    active_match_idx, curr_active_bbox, curr_active_court_pos = match_player_by_position(
                        poses, curr_active_bbox, curr_active_court_pos, K, D, H, exclude_idx=-1
                    )
                    
                    # Only re-identify by label if completely lost AND we had no previous idle tracking
                    if active_match_idx == -1 and curr_active_bbox is None and curr_active_court_pos is None:
                        had_idle_tracking = (curr_idle_bbox is not None or curr_idle_court_pos is not None)
                        if not had_idle_tracking:
                            label_match_idx = identify_player(poses, player_label, K, D, H)
                            if label_match_idx != -1:
                                active_match_idx = label_match_idx
                                curr_active_bbox = unwrap_bbox(poses[label_match_idx]['bbox'])
                                foot_pos = get_foot_position(curr_active_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_active_court_pos = transformed[0]
                
                tracked_active_indices[i] = active_match_idx
                
                # Track Idle: by position, excluding active
                if poses:
                    idle_match_idx, curr_idle_bbox, curr_idle_court_pos = match_player_by_position(
                        poses, curr_idle_bbox, curr_idle_court_pos, K, D, H, exclude_idx=active_match_idx
                    )
                    
                    # If we have active but no idle, and there are multiple poses, assign the other one as idle
                    if active_match_idx != -1 and idle_match_idx == -1 and len(poses) > 1:
                        for p_idx in range(len(poses)):
                            if p_idx != active_match_idx:
                                idle_match_idx = p_idx
                                curr_idle_bbox = unwrap_bbox(poses[p_idx]['bbox'])
                                foot_pos = get_foot_position(curr_idle_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_idle_court_pos = transformed[0]
                                break
                
                tracked_idle_indices[i] = idle_match_idx

            # Generate output with forward-fill for missing frames (max 5 consecutive)
            pose_data_sequence = []
            last_valid_pose = None
            consecutive_lost_frames = 0
            MAX_FORWARD_FILL = 5
            
            # Create output video
            if frames:
                h, w = frames[0].shape[:2]
                clip_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}.mp4"
                clip_path = os.path.join(OUTPUT_DIR, clip_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
            else:
                out = None
            
            for i in range(len(frames)):
                frame = frames[i]
                poses = poses_per_frame[i]
                
                act_idx = tracked_active_indices.get(i, -1)
                idl_idx = tracked_idle_indices.get(i, -1)
                
                # Check if we're forward-filling
                is_forward_filled = False
                forward_filled_pose = None
                if act_idx == -1 and last_valid_pose is not None and consecutive_lost_frames < MAX_FORWARD_FILL:
                    is_forward_filled = True
                    forward_filled_pose = last_valid_pose
                
                # Overlay - pass all poses info if available
                all_poses_info = all_poses_per_frame[i] if (all_poses_per_frame and i < len(all_poses_per_frame)) else None
                vis_frame = draw_overlay(frame, poses, act_idx, idl_idx, debug=DEBUG_MODE, 
                                        all_poses_info=all_poses_info, is_forward_filled=is_forward_filled,
                                        forward_filled_pose=forward_filled_pose)
                if out is not None:
                    out.write(vis_frame)
                
                # Collect pose data with forward-fill (max 5 consecutive frames)
                if act_idx != -1 and act_idx < len(poses):
                    last_valid_pose = poses[act_idx]
                    pose_data_sequence.append(poses[act_idx])
                    consecutive_lost_frames = 0  # Reset counter
                else:
                    # Use previous frame's pose if available and within limit
                    if last_valid_pose is not None and consecutive_lost_frames < MAX_FORWARD_FILL:
                        pose_data_sequence.append(last_valid_pose)
                        consecutive_lost_frames += 1
                    else:
                        pose_data_sequence.append(None)  # Lost for too long or never had valid pose
                        consecutive_lost_frames += 1
            
            if out is not None:
                out.release()
            
            # Save CSV
            pose_csv_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}_pose.csv"
            save_pose_csv(pose_data_sequence, os.path.join(OUTPUT_DIR, pose_csv_filename))

if __name__ == "__main__":
    main()
