import os
import cv2
import numpy as np
import argparse
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import load_fisheye_params, load_perspective_matrix, transform_points, get_foot_position, unwrap_bbox

# Configuration
PARAM_DIR = 'parameters'
FISHEYE_FILE = 'fishcam-fisheye.txt'

# Filtering parameters
CORNER_MARGIN = 0.05  # 5% of image size
BOTTOM_MARGIN = 0.05  # 5% from bottom
MOVEMENT_THRESHOLD = 20.0  # pixels
MIN_FRAMES_FOR_STATIONARY = 5  # frames

def get_calibration(video_name):
    # (Same as in extract_shots.py)
    fisheye_path = os.path.join(PARAM_DIR, FISHEYE_FILE)
    K, D = load_fisheye_params(fisheye_path)
    
    import re
    camera_name = None
    for cam in ['BO01', 'BO02', 'LU01', 'LU02']:
        if cam in video_name: 
            camera_name = cam
            break
            
    if not camera_name:
        for prefix in ['BO', 'LU']:
            match = re.search(f"{prefix}-(\d+)", video_name)
            if match:
                num = int(match.group(1))
                camera_name = f"{prefix}{num:02d}"
                break

    H = None
    if camera_name:
        perspective_path = os.path.join(PARAM_DIR, f"{camera_name}-perspective.txt")
        H = load_perspective_matrix(perspective_path)
        print(f"Loaded calibration for {camera_name}")
    else:
        print(f"Warning: Could not determine camera name from {video_name}")

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
        # Otherwise, it's filtered out
    
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

def draw_court_debug(img, poses, K, D, H, img_width=None, img_height=None, show_filtered_info=True, 
                    filtered_pose_indices=None, flickering_pose_indices=None):
    """
    Draws a top-down view of the court with player positions.
    
    Args:
        filtered_pose_indices: Set of pose indices that should be shown in orange (filtered)
        flickering_pose_indices: Set of pose indices that should be shown in black (flickering)
    """
    if filtered_pose_indices is None:
        filtered_pose_indices = set()
    if flickering_pose_indices is None:
        flickering_pose_indices = set()
    
    h, w = img.shape[:2]
    
    # Create debug panel (width = w/2)
    debug_w = int(w / 2)
    debug_h = h
    debug_img = np.zeros((debug_h, debug_w, 3), dtype=np.uint8)
    
    # Define Court in debug view
    # Real court: 0-10m (x), 0-20m (y)
    # Map to debug_img:
    # Margin: 20px
    margin = 50
    scale_x = (debug_w - 2 * margin) / 10.0
    scale_y = (debug_h - 2 * margin) / 20.0
    # Keep aspect ratio?
    scale = min(scale_x, scale_y)
    
    origin_x = margin
    origin_y = debug_h - margin # Bottom-left is (0,0) usually? 
    # Actually padel coordinates: let's assume 0,0 is bottom-left
    
    def to_debug(mx, my):
        px = int(origin_x + mx * scale)
        py = int(origin_y - my * scale) # Y up in meters, Y down in image
        return px, py

    # Draw Court Boundaries (0,0) to (10,20)
    c00 = to_debug(0, 0)
    c100 = to_debug(10, 0)
    c1020 = to_debug(10, 20)
    c020 = to_debug(0, 20)
    
    cv2.rectangle(debug_img, (c00[0], c020[1]), (c100[0], c00[1]), (50, 50, 50), -1) # Court floor
    cv2.rectangle(debug_img, (c00[0], c020[1]), (c100[0], c00[1]), (255, 255, 255), 2) # Boundary
    
    # Draw Net (y=10)
    net_l = to_debug(0, 10)
    net_r = to_debug(10, 10)
    cv2.line(debug_img, net_l, net_r, (200, 200, 200), 2)
    
    # Draw Poses
    for i, pose in enumerate(poses):
        bbox = unwrap_bbox(pose['bbox'])
        if len(bbox) < 4: continue
        
        # Extract confidence scores
        # Bbox score (if available, might be in bbox[4] or as separate field)
        bbox_score = None
        if len(bbox) >= 5:
            bbox_score = bbox[4]
        elif 'bbox_score' in pose:
            bbox_score = pose['bbox_score']
        elif 'score' in pose:
            bbox_score = pose['score']
        
        # Keypoint scores
        kp_scores = None
        if 'keypoint_scores' in pose:
            kp_scores = pose['keypoint_scores']
        elif 'keypoints' in pose and len(pose['keypoints']) > 0:
            # Check if keypoints include scores (format: [x, y, score])
            first_kp = pose['keypoints'][0]
            if len(first_kp) >= 3:
                kp_scores = [kp[2] if len(kp) >= 3 else 0.0 for kp in pose['keypoints']]
        
        # Calculate average keypoint score
        avg_kp_score = None
        max_kp_score = None
        if kp_scores is not None and len(kp_scores) > 0:
            valid_scores = [s for s in kp_scores if s > 0]
            if valid_scores:
                avg_kp_score = np.mean(valid_scores)
                max_kp_score = np.max(valid_scores)
        
        foot_pos = get_foot_position(bbox)
        
        # Check filtering criteria
        is_near_corner_bottom = False
        if img_width is not None and img_height is not None:
            is_near_corner_bottom = is_pose_near_corners_or_bottom(bbox, img_width, img_height, 
                                                                   CORNER_MARGIN, BOTTOM_MARGIN)
        
        # Check if this pose is marked as filtered (stationary or corner/bottom)
        is_filtered = (i in filtered_pose_indices) or is_near_corner_bottom
        is_flickering = (i in flickering_pose_indices)
        
        # Transform
        if H is not None:
            transformed = transform_points([foot_pos], K, D, H)
            if len(transformed) > 0:
                tx, ty = transformed[0]
                
                # Check valid (matching extract_shots.py bounds)
                is_in_court = (-1.1 <= tx <= 10.7 and -0.7 <= ty <= 20.7)
                is_low_enough = (foot_pos[1] > 0.25 * h) # Note: 0.25 threshold
                
                # Determine color based on validity and filtering
                if is_flickering:
                    color = (0, 0, 0)  # Black - flickering (appears in < 80% of frames)
                elif is_filtered:
                    color = (0, 165, 255)  # Orange - filtered (corner/bottom or stationary)
                elif is_in_court and is_low_enough:
                    color = (0, 255, 0)  # Green - valid
                else:
                    color = (0, 0, 255)  # Red - invalid
                
                # Draw on Debug Map
                dpx, dpy = to_debug(tx, ty)
                cv2.circle(debug_img, (dpx, dpy), 8, color, -1)
                cv2.putText(debug_img, f"P{i}", (dpx+10, dpy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Text Info with confidence scores and filtering status
                info_y = 30 + i * 30
                if is_flickering:
                    status = "FLICKERING"
                    reason = ["<80% presence"]
                elif is_filtered:
                    if i in filtered_pose_indices:
                        status = "FILTERED"
                        reason = ["Stationary"]
                    else:
                        status = "FILTERED"
                        reason = ["Corner/Bottom"]
                elif is_in_court and is_low_enough:
                    status = "OK"
                    reason = []
                else:
                    status = "FAIL"
                    reason = []
                    if not is_in_court: reason.append(f"Out: {tx:.1f},{ty:.1f}")
                    if not is_low_enough: reason.append(f"High: y={foot_pos[1]}/{h}")
                
                # Build confidence string
                conf_parts = []
                if bbox_score is not None:
                    conf_parts.append(f"bbox:{bbox_score:.2f}")
                if avg_kp_score is not None:
                    conf_parts.append(f"kp_avg:{avg_kp_score:.2f}")
                if max_kp_score is not None:
                    conf_parts.append(f"kp_max:{max_kp_score:.2f}")
                conf_str = " | ".join(conf_parts) if conf_parts else "no_conf"
                
                info_text = f"P{i}: {status} {' '.join(reason)} | {conf_str}"
                cv2.putText(debug_img, info_text, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw on Main Image with confidence
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Build label with confidence and filtering status
                label_parts = [f"P{i} ({tx:.1f}, {ty:.1f})"]
                if is_flickering:
                    label_parts.append("FLICKERING")
                elif is_filtered:
                    label_parts.append("FILTERED")
                if bbox_score is not None:
                    label_parts.append(f"bbox:{bbox_score:.2f}")
                if avg_kp_score is not None:
                    label_parts.append(f"kp:{avg_kp_score:.2f}")
                label = " | ".join(label_parts)
                
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
    return np.hstack((img, debug_img))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Center frame (will extract 15 before and 14 after)')
    args = parser.parse_args()
    
    video_name = os.path.basename(args.video_path)
    K, D, H = get_calibration(video_name)
    
    # Init model
    model_config = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
    model_ckpt = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
    inferencer = MMPoseInferencer(pose2d=model_config, pose2d_weights=model_ckpt)
    
    # Frame range: 15 before, center, 14 after = 30 frames (matching extract_shots.py)
    center_frame = args.frame
    start_frame = max(0, center_frame - 15)
    duration = 30
    
    cap = cv2.VideoCapture(args.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    out_path = f"debug_{video_name}_{center_frame}.mp4"
    print(f"Extracting frames {start_frame} to {start_frame + duration - 1} (center: {center_frame})")
    print(f"Writing debug video to {out_path}")
    
    # First pass: collect all frames and poses
    frames = []
    all_predictions_per_frame = []
    
    for _ in range(duration):
        ret, frame = cap.read()
        if not ret: break
        
        frames.append(frame)
        
        # Inference
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        predictions = result['predictions'][0]
        all_predictions_per_frame.append(predictions if predictions else [])
    
    cap.release()
    
    # Track which poses are filtered (for visualization)
    # We'll show all poses but mark filtered ones in orange
    filtered_poses_per_frame = []  # List of sets, each set contains indices of filtered poses in that frame
    
    # Apply filtering: corner and bottom+stationary
    # First, filter top corners per frame and track which ones
    corner_filtered_per_frame = []
    filtered_predictions_per_frame = []
    for frame_idx, predictions in enumerate(all_predictions_per_frame):
        filtered_frame_predictions = []
        corner_indices = set()
        for pose_idx, pose in enumerate(predictions):
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4:
                continue
            # Check if top corner filtered (not bottom - that's handled with stationary)
            foot_pos = get_foot_position(bbox)
            foot_x, foot_y = foot_pos
            corner_threshold_x = img_width * CORNER_MARGIN
            corner_threshold_y = img_height * CORNER_MARGIN
            
            is_top_corner = (foot_x < corner_threshold_x and foot_y < corner_threshold_y) or \
                           (foot_x > img_width * (1 - CORNER_MARGIN) and foot_y < corner_threshold_y)
            
            if is_top_corner:
                corner_indices.add(pose_idx)
            else:
                filtered_frame_predictions.append(pose)
        filtered_predictions_per_frame.append(filtered_frame_predictions)
        corner_filtered_per_frame.append(corner_indices)
    
    # Then, filter poses that are BOTH near bottom AND stationary
    filtered_predictions_per_frame = filter_stationary_poses(
        filtered_predictions_per_frame, 
        movement_threshold=MOVEMENT_THRESHOLD, 
        min_frames=MIN_FRAMES_FOR_STATIONARY,
        img_height=img_height,
        bottom_margin=BOTTOM_MARGIN,
        filter_bottom_stationary_only=True
    )
    
    # Apply flickering filter (poses that appear in < 80% of frames)
    filtered_predictions_per_frame, flickering_tracks = filter_flickering_poses(
        filtered_predictions_per_frame, min_presence_ratio=0.8
    )
    
    # Build mapping of filtered and flickering poses for visualization
    # Track which poses are filtered (corner or bottom+stationary) and flickering
    flickering_poses_per_frame = [set() for _ in range(len(all_predictions_per_frame))]
    
    # Map flickering tracks back to original pose indices
    for flickering_track in flickering_tracks:
        for frame_idx in flickering_track['frames']:
            if frame_idx < len(flickering_poses_per_frame):
                # Find matching pose in original predictions for this frame
                # We'll mark all poses in frames where flickering occurred
                # This is approximate but should work for visualization
                pass  # Will be handled below
    
    for frame_idx, predictions in enumerate(all_predictions_per_frame):
        filtered_indices = set()
        # Add corner filtered
        filtered_indices.update(corner_filtered_per_frame[frame_idx])
        
        # Track stationary filtered by comparing before/after stationary filtering
        # We need to compare the list after corner/bottom filtering vs after stationary filtering
        # But we need to match poses. Let's use a simpler approach:
        # After corner/bottom filtering, we have some poses. After stationary filtering, some are removed.
        # We'll identify which ones by matching bbox positions
        
        # Get poses after corner filtering (but before bottom+stationary)
        after_corner = []
        for pose_idx, pose in enumerate(predictions):
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4:
                continue
            # Check if it's a top corner (not bottom)
            foot_pos = get_foot_position(bbox)
            foot_x, foot_y = foot_pos
            corner_threshold_x = img_width * CORNER_MARGIN
            corner_threshold_y = img_height * CORNER_MARGIN
            is_top_corner = (foot_x < corner_threshold_x and foot_y < corner_threshold_y) or \
                           (foot_x > img_width * (1 - CORNER_MARGIN) and foot_y < corner_threshold_y)
            if not is_top_corner:
                after_corner.append((pose_idx, pose))
        
        # Get poses after stationary filtering
        after_stationary = filtered_predictions_per_frame[frame_idx] if frame_idx < len(filtered_predictions_per_frame) else []
        
        # Match poses to find which ones were filtered as bottom+stationary or flickering
        # Simple matching by bbox center proximity
        for orig_idx, orig_pose in after_corner:
            orig_bbox = unwrap_bbox(orig_pose['bbox'])
            if len(orig_bbox) < 4:
                continue
            orig_center = ((orig_bbox[0] + orig_bbox[2]) / 2, (orig_bbox[1] + orig_bbox[3]) / 2)
            
            found_match = False
            for filtered_pose in after_stationary:
                filtered_bbox = unwrap_bbox(filtered_pose['bbox'])
                if len(filtered_bbox) < 4:
                    continue
                filtered_center = ((filtered_bbox[0] + filtered_bbox[2]) / 2, 
                                  (filtered_bbox[1] + filtered_bbox[3]) / 2)
                
                dist = np.sqrt((orig_center[0] - filtered_center[0])**2 + 
                              (orig_center[1] - filtered_center[1])**2)
                if dist < 50:  # Same pose if centers within 50 pixels
                    found_match = True
                    break
            
            # If not found in filtered list, it was filtered as bottom+stationary or flickering
            if not found_match:
                # Check if it's flickering by checking if this frame is in any flickering track
                is_flickering = False
                for flickering_track in flickering_tracks:
                    if frame_idx in flickering_track['frames']:
                        # Check if position matches (approximate)
                        for f_idx in flickering_track['frames']:
                            if f_idx == frame_idx:
                                is_flickering = True
                                break
                        if is_flickering:
                            break
                
                if is_flickering:
                    flickering_poses_per_frame[frame_idx].add(orig_idx)
                else:
                    filtered_indices.add(orig_idx)
        
        filtered_poses_per_frame.append(filtered_indices)
    
    # Second pass: visualize all predictions, marking filtered ones in orange
    vw = None
    for frame_idx, frame in enumerate(frames):
        # Show ALL original predictions
        predictions = all_predictions_per_frame[frame_idx] if frame_idx < len(all_predictions_per_frame) else []
        filtered_indices = filtered_poses_per_frame[frame_idx] if frame_idx < len(filtered_poses_per_frame) else set()
        
        # Visualize all poses, marking filtered ones in orange and flickering ones in black
        flickering_indices = flickering_poses_per_frame[frame_idx] if frame_idx < len(flickering_poses_per_frame) else set()
        vis = draw_court_debug(frame, predictions, K, D, H, img_width, img_height, 
                              show_filtered_info=True, filtered_pose_indices=filtered_indices,
                              flickering_pose_indices=flickering_indices)
        
        if vw is None:
            h, w = vis.shape[:2]
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            
        vw.write(vis)
    
    if vw: vw.release()
    print("Done.")

if __name__ == "__main__":
    main()

