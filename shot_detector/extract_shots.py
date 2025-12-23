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

def is_pose_near_bottom(bbox, img_height, bottom_margin=0.05):
    """
    Checks if a pose is near the bottom of the image.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        img_height: Image height
        bottom_margin: Margin from bottom as fraction of image height (default 5%)
    
    Returns:
        True if pose is near bottom, False otherwise
    """
    if len(bbox) < 4:
        return False
    
    foot_pos = get_foot_position(bbox)
    foot_y = foot_pos[1]
    
    # Check if near bottom (within bottom_margin of bottom edge)
    return foot_y > img_height * (1 - bottom_margin)

def is_pose_near_corners_or_bottom(bbox, img_width, img_height, corner_margin=0.05, bottom_margin=0.05):
    """
    Checks if a pose is near the corners or bottom of the image.
    These are often false positives from reflections or static objects.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height
        corner_margin: Margin as fraction of image size (default 5%)
        bottom_margin: Margin from bottom as fraction of image height (default 5%)
    
    Returns:
        True if pose is near corners or bottom, False otherwise
    """
    if len(bbox) < 4:
        return False
    
    x1, y1, x2, y2 = bbox[:4]
    foot_pos = get_foot_position(bbox)
    foot_x, foot_y = foot_pos
    
    # Check if near bottom (within bottom_margin of bottom edge)
    if foot_y > img_height * (1 - bottom_margin):
        return True
    
    # Check if near corners
    corner_threshold_x = img_width * corner_margin
    corner_threshold_y = img_height * corner_margin
    
    # Bottom-left corner
    if foot_x < corner_threshold_x and foot_y > img_height * (1 - corner_margin):
        return True
    
    # Bottom-right corner
    if foot_x > img_width * (1 - corner_margin) and foot_y > img_height * (1 - corner_margin):
        return True
    
    # Top-left corner (less common but possible)
    if foot_x < corner_threshold_x and foot_y < corner_threshold_y:
        return True
    
    # Top-right corner
    if foot_x > img_width * (1 - corner_margin) and foot_y < corner_threshold_y:
        return True
    
    return False

def filter_stationary_poses(poses_per_frame, movement_threshold=20.0, min_frames=5, 
                           img_height=None, bottom_margin=0.05, filter_bottom_stationary_only=False):
    """
    Filters poses that barely move across frames (likely false positives).
    If filter_bottom_stationary_only is True, only filters poses that are BOTH near bottom AND stationary.
    
    Args:
        poses_per_frame: List of lists, where each inner list contains poses for a frame
        movement_threshold: Minimum pixel movement required (default 20 pixels)
        min_frames: Minimum number of frames the pose must appear in to be considered stationary (default 5)
        img_height: Image height (required if filter_bottom_stationary_only is True)
        bottom_margin: Margin from bottom as fraction of image height (default 5%)
        filter_bottom_stationary_only: If True, only filter poses that are near bottom AND stationary
    
    Returns:
        List of lists with stationary poses removed
    """
    if not poses_per_frame or len(poses_per_frame) < min_frames:
        return poses_per_frame
    
    # Build a list of all poses with their frame indices and positions
    pose_tracks = []  # List of (frame_idx, foot_pos, pose, is_near_bottom)
    
    for frame_idx, frame_poses in enumerate(poses_per_frame):
        for pose in frame_poses:
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4:
                continue
            foot_pos = get_foot_position(bbox)
            # Check if near bottom (if filtering bottom+stationary only)
            is_near_bottom = False
            if filter_bottom_stationary_only and img_height is not None:
                is_near_bottom = is_pose_near_bottom(bbox, img_height, bottom_margin)
            pose_tracks.append((frame_idx, foot_pos, pose, is_near_bottom))
    
    # Group poses into tracks by proximity
    # Simple approach: poses in nearby frames with similar positions are the same track
    tracks = []  # List of lists of (frame_idx, foot_pos, pose, is_near_bottom)
    match_distance = 100  # Pixels
    
    for frame_idx, foot_pos, pose, is_near_bottom in pose_tracks:
        matched = False
        for track in tracks:
            # Check if this pose matches any pose in the track
            for track_frame_idx, track_foot_pos, _, _ in track:
                dist = np.sqrt((foot_pos[0] - track_foot_pos[0])**2 + 
                              (foot_pos[1] - track_foot_pos[1])**2)
                # Match if close in position and within reasonable frame distance
                if dist < match_distance and abs(frame_idx - track_frame_idx) <= min_frames:
                    track.append((frame_idx, foot_pos, pose, is_near_bottom))
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            tracks.append([(frame_idx, foot_pos, pose, is_near_bottom)])
    
    # Filter out stationary tracks
    filtered_poses_per_frame = [[] for _ in range(len(poses_per_frame))]
    filtered_count = 0
    
    for track in tracks:
        if len(track) < min_frames:
            # Too few appearances, keep it (might be valid but brief)
            for frame_idx, _, pose, _ in track:
                filtered_poses_per_frame[frame_idx].append(pose)
            continue
        
        # Calculate maximum movement in this track
        positions = [foot_pos for _, foot_pos, _, _ in track]
        max_movement = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                movement = np.sqrt((positions[i][0] - positions[j][0])**2 +
                                  (positions[i][1] - positions[j][1])**2)
                max_movement = max(max_movement, movement)
        
        # Check if track is near bottom (if any pose in track is near bottom, consider it near bottom)
        track_is_near_bottom = any(is_near_bottom for _, _, _, is_near_bottom in track)
        
        # Decide whether to filter:
        # - If filter_bottom_stationary_only: only filter if BOTH stationary AND near bottom
        # - Otherwise: filter if stationary
        should_filter = False
        if filter_bottom_stationary_only:
            # Only filter if stationary AND near bottom
            if max_movement < movement_threshold and track_is_near_bottom:
                should_filter = True
        else:
            # Filter if stationary (regardless of position)
            if max_movement < movement_threshold:
                should_filter = True
        
        # Keep track if it moves enough or doesn't meet filter criteria
        if not should_filter:
            for frame_idx, _, pose, _ in track:
                filtered_poses_per_frame[frame_idx].append(pose)
        else:
            # Track is filtered out
            filtered_count += len(track)
    
    return filtered_poses_per_frame

def filter_flickering_poses(poses_per_frame, min_presence_ratio=0.8):
    """
    Filters poses that flicker (don't appear in at least min_presence_ratio of frames).
    
    Args:
        poses_per_frame: List of lists, where each inner list contains poses for a frame
        min_presence_ratio: Minimum ratio of frames the pose must appear in (default 0.8 = 80%)
    
    Returns:
        Tuple of (filtered_poses_per_frame, flickering_tracks)
        flickering_tracks: List of track info for poses that were filtered as flickering
    """
    if not poses_per_frame or len(poses_per_frame) == 0:
        return poses_per_frame, []
    
    total_frames = len(poses_per_frame)
    min_frames_required = int(total_frames * min_presence_ratio)
    
    # Build tracks (same as in filter_stationary_poses)
    pose_tracks = []  # List of (frame_idx, foot_pos, pose)
    
    for frame_idx, frame_poses in enumerate(poses_per_frame):
        for pose in frame_poses:
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4:
                continue
            foot_pos = get_foot_position(bbox)
            pose_tracks.append((frame_idx, foot_pos, pose))
    
    # Group poses into tracks by proximity
    tracks = []
    match_distance = 100  # Pixels
    
    for frame_idx, foot_pos, pose in pose_tracks:
        matched = False
        for track in tracks:
            for track_frame_idx, track_foot_pos, _ in track:
                dist = np.sqrt((foot_pos[0] - track_foot_pos[0])**2 + 
                              (foot_pos[1] - track_foot_pos[1])**2)
                if dist < match_distance:
                    track.append((frame_idx, foot_pos, pose))
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            tracks.append([(frame_idx, foot_pos, pose)])
    
    # Filter out flickering tracks
    filtered_poses_per_frame = [[] for _ in range(len(poses_per_frame))]
    flickering_tracks = []
    
    for track in tracks:
        presence_ratio = len(track) / total_frames
        
        if presence_ratio >= min_presence_ratio:
            # Keep track - appears in enough frames
            for frame_idx, _, pose in track:
                filtered_poses_per_frame[frame_idx].append(pose)
        else:
            # Flickering - mark for visualization but don't include in filtered output
            flickering_tracks.append({
                'frames': [f_idx for f_idx, _, _ in track],
                'presence_ratio': presence_ratio
            })
    
    return filtered_poses_per_frame, flickering_tracks

def is_pose_valid(pose, K, D, H, img_height=None, img_width=None):
    """
    Checks if a pose is within the imaginary field (0-10, 0-20).
    Also filters background players (y < 30% of image height).
    Filters poses near corners/bottom of image.
    """
    bbox = unwrap_bbox(pose['bbox'])
    if len(bbox) < 4:
        return False
    
    # 1. Pixel-based filtering (y > 30% of height)
    if img_height is not None:
        foot_y = get_foot_position(bbox)[1]
        if foot_y < 0.25 * img_height:
             return False # Too far up (background players)
        
        # 2. Filter poses near corners (bottom filtering is done in combination with stationary)
        if img_width is not None:
            # Only filter top corners, not bottom (bottom is handled with stationary filtering)
            corner_margin = 0.05
            corner_threshold_x = img_width * corner_margin
            corner_threshold_y = img_height * corner_margin
            foot_pos = get_foot_position(bbox)
            foot_x, foot_y = foot_pos
            
            # Check top corners only (not bottom corners)
            # Top-left corner
            if foot_x < corner_threshold_x and foot_y < corner_threshold_y:
                return False
            # Top-right corner
            if foot_x > img_width * (1 - corner_margin) and foot_y < corner_threshold_y:
                return False

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
    if -1.1 <= tx <= 10.7 and -0.7 <= ty <= 20.7:
        return True
        
    return False

def extract_clip_and_pose(video_path, start_frame, duration, inferencer, K=None, D=None, H=None, return_all_poses=False):
    """
    Extracts frames, runs pose estimation, and returns data.
    Filters poses based on calibration if provided.
    Also filters poses near corners/bottom and stationary poses.
    
    Args:
        return_all_poses: If True, returns all poses (filtered and unfiltered) for debugging
    
    Returns:
        If return_all_poses: (frames, poses_per_frame, all_poses_per_frame, filtering_info)
        filtering_info: Dict with 'filtered_per_frame' and 'flickering_per_frame' lists
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
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
                    is_valid = is_pose_valid(instance, K, D, H, img_height=img_height, img_width=img_width)
                    frame_all_poses.append((instance, is_valid))
                
                # Filter here (corner/bottom filtering included)
                if is_pose_valid(instance, K, D, H, img_height=img_height, img_width=img_width):
                    frame_poses.append(instance)
                
        poses_per_frame.append(frame_poses)
        if return_all_poses:
            all_poses_per_frame.append(frame_all_poses)

    cap.release()
    
    # Debug: count poses before filtering
    total_poses_before = sum(len(p) for p in poses_per_frame)
    
    # Save original poses before filtering for visualization
    original_poses_per_frame = [p.copy() for p in poses_per_frame]
    
    # Apply stationary pose filtering (removes poses that are near bottom AND barely move)
    poses_per_frame_after_stationary = filter_stationary_poses(poses_per_frame, movement_threshold=20.0, min_frames=5,
                                             img_height=img_height, bottom_margin=0.05, 
                                             filter_bottom_stationary_only=True)
    
    # Debug: count poses after stationary filtering
    total_poses_after_stationary = sum(len(p) for p in poses_per_frame_after_stationary)
    
    # Apply flickering pose filtering (removes poses that appear in < 80% of frames)
    poses_per_frame_after_flickering, flickering_tracks = filter_flickering_poses(poses_per_frame_after_stationary, min_presence_ratio=0.8)
    
    # Debug: count poses after flickering filtering
    total_poses_after_flickering = sum(len(p) for p in poses_per_frame_after_flickering)
    
    # Debug output to verify filtering is working
    if DEBUG_MODE and (total_poses_before > total_poses_after_flickering):
        print(f"  Filtering applied: {total_poses_before} -> {total_poses_after_stationary} (after stationary) -> {total_poses_after_flickering} (after flickering)")
        print(f"    Removed {total_poses_before - total_poses_after_stationary} poses (stationary+bottom)")
        print(f"    Removed {total_poses_after_stationary - total_poses_after_flickering} poses (flickering)")
    
    # Build filtering info for visualization: track which poses were filtered
    filtering_info = {
        'filtered_per_frame': [set() for _ in range(len(original_poses_per_frame))],
        'flickering_per_frame': [set() for _ in range(len(original_poses_per_frame))]
    }
    
    # Map filtered poses back to original indices by matching bboxes
    for frame_idx in range(len(original_poses_per_frame)):
        original_poses = original_poses_per_frame[frame_idx]
        after_stationary = poses_per_frame_after_stationary[frame_idx] if frame_idx < len(poses_per_frame_after_stationary) else []
        after_flickering = poses_per_frame_after_flickering[frame_idx] if frame_idx < len(poses_per_frame_after_flickering) else []
        
        # Find poses filtered by stationary
        for orig_idx, orig_pose in enumerate(original_poses):
            orig_bbox = unwrap_bbox(orig_pose['bbox'])
            if len(orig_bbox) < 4:
                continue
            orig_center = ((orig_bbox[0] + orig_bbox[2]) / 2, (orig_bbox[1] + orig_bbox[3]) / 2)
            
            found_in_stationary = False
            for filtered_pose in after_stationary:
                filtered_bbox = unwrap_bbox(filtered_pose['bbox'])
                if len(filtered_bbox) < 4:
                    continue
                filtered_center = ((filtered_bbox[0] + filtered_bbox[2]) / 2, 
                                  (filtered_bbox[1] + filtered_bbox[3]) / 2)
                dist = np.sqrt((orig_center[0] - filtered_center[0])**2 + 
                              (orig_center[1] - filtered_center[1])**2)
                if dist < 50:
                    found_in_stationary = True
                    break
            
            if not found_in_stationary:
                filtering_info['filtered_per_frame'][frame_idx].add(orig_idx)
            else:
                # Check if it's in flickering (was in stationary but not in flickering)
                found_in_flickering = False
                for flickering_pose in after_flickering:
                    flickering_bbox = unwrap_bbox(flickering_pose['bbox'])
                    if len(flickering_bbox) < 4:
                        continue
                    flickering_center = ((flickering_bbox[0] + flickering_bbox[2]) / 2, 
                                        (flickering_bbox[1] + flickering_bbox[3]) / 2)
                    dist = np.sqrt((orig_center[0] - flickering_center[0])**2 + 
                                  (orig_center[1] - flickering_center[1])**2)
                    if dist < 50:
                        found_in_flickering = True
                        break
                
                if not found_in_flickering:
                    filtering_info['flickering_per_frame'][frame_idx].add(orig_idx)
    
    # Use filtered poses for actual processing
    poses_per_frame = poses_per_frame_after_flickering
    
    if return_all_poses:
        return frames, poses_per_frame, all_poses_per_frame, filtering_info
    return frames, poses_per_frame

def unwrap_bbox(bbox):
    """
    Helper to unwrap nested bbox structure ([x1,y1,x2,y2],) -> [x1,y1,x2,y2]
    """
    if len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
        return bbox[0]
    return bbox

def draw_overlay(frame, poses, active_idx, idle_idx, debug=False, all_poses_info=None, 
                is_forward_filled=False, forward_filled_pose=None, filtering_info=None):
    """
    Draws bounding boxes and labels on the frame.
    
    Args:
        debug: If True, shows all poses with indices and debug info
        all_poses_info: List of (pose, is_valid) tuples for all detected poses (including filtered ones)
        is_forward_filled: If True, active player is forward-filled (show in blue)
        forward_filled_pose: The pose to draw when forward-filling (if None, uses last pose in poses)
        filtering_info: Dict with 'filtered_per_frame' and 'flickering_per_frame' sets of pose indices
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
            
            # Check filtering status
            is_filtered = False
            is_flickering = False
            if filtering_info is not None:
                # Note: i is the index in all_poses_info, which should match the original pose index
                # We need to get the frame index - but we don't have it here. 
                # For now, we'll check if this pose matches any filtered pose by bbox
                # This is approximate but should work
                this_bbox = unwrap_bbox(pose['bbox'])
                if len(this_bbox) >= 4:
                    # Try to match with filtered/flickering poses by checking all frames
                    # This is a simplified approach - ideally we'd pass frame_idx
                    pass  # Will handle below with bbox matching
            
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
                
                # Check if this pose was filtered (stationary/bottom) or flickering
                if filtering_info is not None:
                    # filtering_info is now per-frame: {'filtered': set(), 'flickering': set()}
                    if i in filtering_info.get('flickering', set()):
                        is_flickering = True
                    elif i in filtering_info.get('filtered', set()):
                        is_filtered = True
                
                # Determine color based on status
                if is_flickering:
                    color = (0, 0, 0)  # Black - flickering
                    label = f"P{i} FLICKERING"
                elif is_filtered:
                    color = (0, 165, 255)  # Orange - filtered (stationary/bottom)
                    label = f"P{i} FILTERED"
                elif tracked_as == 'active':
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
                color = (255, 0, 255)  # Magenta - invalid (initial validation failed)
                label = f"P{i} INVALID"
            
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
                result = extract_clip_and_pose(
                    video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=True
                )
                frames, poses_per_frame, all_poses_per_frame, filtering_info = result
            else:
                frames, poses_per_frame = extract_clip_and_pose(
                    video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=False
                )
                all_poses_per_frame = None
                filtering_info = None
            
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
                # Get filtering info for this frame
                frame_filtering_info = None
                if filtering_info is not None and i < len(filtering_info.get('filtered_per_frame', [])):
                    frame_filtering_info = {
                        'filtered': filtering_info['filtered_per_frame'][i] if i < len(filtering_info['filtered_per_frame']) else set(),
                        'flickering': filtering_info['flickering_per_frame'][i] if i < len(filtering_info['flickering_per_frame']) else set()
                    }
                vis_frame = draw_overlay(frame, poses, act_idx, idl_idx, debug=DEBUG_MODE, 
                                        all_poses_info=all_poses_info, is_forward_filled=is_forward_filled,
                                        forward_filled_pose=forward_filled_pose, filtering_info=frame_filtering_info)
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
