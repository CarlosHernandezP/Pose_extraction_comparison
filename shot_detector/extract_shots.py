import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import parse_shot_csv, get_video_path, identify_player, get_idle_player

# Configuration
SHOTS_CSV_DIR = '/home/daniele/shots_csvs/'
VIDEOS_DIR = '/home/daniele/videos/'
OUTPUT_DIR = 'shot_detector/data/'
MODEL_CONFIG = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
MODEL_CHECKPOINT = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_mmpose():
    if not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Config not found: {MODEL_CONFIG}")
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CHECKPOINT}")
    
    print(f"Initializing MMPose with {MODEL_CONFIG} on {DEVICE}...")
    return MMPoseInferencer(
        pose2d=MODEL_CONFIG,
        pose2d_weights=MODEL_CHECKPOINT,
        device=DEVICE
    )

def extract_clip_and_pose(video_path, start_frame, duration, inferencer):
    """
    Extracts frames, runs pose estimation, and returns data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    poses_per_frame = []
    
    for _ in range(duration):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Run inference on single frame
        # inferencer returns a generator
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        
        # Extract predictions
        # structure: result['predictions'][0][0]['keypoints']... 
        # Actually standard mmpose output format check needed
        # result['predictions'] is a list of results (one per image provided)
        # Each result usually has 'instances' or similar keys.
        # Let's assume standard MMPose output structure
        
        predictions = result['predictions'][0]
        # In newer MMPose, predictions might be an object or dict. 
        # Usually it has 'keypoints', 'bbox', 'bbox_score', 'keypoint_scores'
        
        # We need to normalize this into a list of dicts for our utils
        frame_poses = []
        
        if predictions:
            # prediction is often a list of dicts if multiple people
            # OR a dict with keys containing lists.
            # Let's inspect what MMPoseInferencer returns exactly for RTMO
            # Typically: [{'bbox': [...], 'keypoints': [...], ...}, ...]
            
            # If it's the structure from high-level apis:
            for instance in predictions:
                frame_poses.append(instance)
                
        poses_per_frame.append(frame_poses)

    cap.release()
    return frames, poses_per_frame

def unwrap_bbox(bbox):
    """
    Helper to unwrap nested bbox structure ([x1,y1,x2,y2],) -> [x1,y1,x2,y2]
    """
    if len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
        return bbox[0]
    return bbox

def draw_overlay(frame, poses, active_idx, idle_idx):
    """
    Draws bounding boxes and labels on the frame.
    """
    img = frame.copy()
    
    for i, pose in enumerate(poses):
        bbox = unwrap_bbox(pose['bbox']) # Unwrap here
        if len(bbox) < 4: continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        color = (255, 255, 255) # Default white
        label = ""
        
        if i == active_idx:
            color = (0, 255, 0) # Green for Active
            label = "ACTIVE"
        elif i == idle_idx:
            color = (0, 0, 255) # Red for Idle
            label = "IDLE"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if label:
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        # Draw Keypoints (simplified)
        if 'keypoints' in pose:
            kpts = pose['keypoints']
            for kp in kpts:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0: # valid keypoint
                    cv2.circle(img, (kx, ky), 3, color, -1)

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
    
    # List CSVs
    csv_files = [f for f in os.listdir(SHOTS_CSV_DIR) if f.endswith('.csv')]
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        csv_path = os.path.join(SHOTS_CSV_DIR, csv_file)
        # print(f"Processing {csv_file}...") # tqdm handles this
        
        df = parse_shot_csv(csv_path)
        if df.empty:
            continue
            
        video_path = get_video_path(csv_path, VIDEOS_DIR)
        if not video_path:
            tqdm.write(f"Video not found for {csv_file}")
            continue
            
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
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
            
            # print(f"  > Shot: {shot_type} at frame {center_frame} ({player_label})")
            
            # Extract data
            frames, poses_per_frame = extract_clip_and_pose(video_path, start_frame, duration, inferencer)
            
            if not frames or not poses_per_frame:
                tqdm.write(f"    Failed to extract frames for shot {shot_type} at {center_frame}")
                continue
                
            # Identify players in the CENTER frame (approx index 15)
            # Use the poses from the center frame to decide which index tracks which player
            # Note: Tracking across frames is not guaranteed by simple per-frame inference.
            # However, for 1 second clip, indexes often stay stable if people don't cross.
            # Better approach: Find player in center frame, then find closest match in adjacent frames.
            # For now, let's assume index stability or re-identify per frame?
            # Re-identifying per frame based on spatial consistency is safer.
            
            # Let's pick the active player in the center frame
            center_idx_in_clip = min(15, len(frames) - 1)
            center_poses = poses_per_frame[center_idx_in_clip]
            
            active_idx_initial = identify_player(center_poses, player_label)
            
            if active_idx_initial == -1:
                tqdm.write(f"    Could not find player '{player_label}' in center frame for shot {shot_type} at {center_frame}")
                continue
                
            idle_idx_initial = get_idle_player(center_poses, active_idx_initial)
            
            # Track players across the clip (simple distance tracking)
            active_poses_sequence = []
            
            # Output Video writer
            clip_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}.mp4"
            clip_path = os.path.join(OUTPUT_DIR, clip_filename)
            
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
            
            prev_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            
            # We need to re-process the list to ensure we track the correct person
            # Iterate through all frames
            for i in range(len(frames)):
                frame = frames[i]
                current_poses = poses_per_frame[i]
                
                # Find active player in this frame (closest to previous bbox center)
                # For first frame, we might work backwards from center?
                # Or just assume linear motion.
                # Let's process forward from center, and backward from center.
                pass 

            # SIMPLIFIED TRACKING STRATEGY:
            # 1. Identify active/idle in EVERY frame using the label logic (Left/Right/etc).
            #    Issue: Players move. 'Left' player might become 'Right' if they cross?
            #    But for a shot, they usually stay in their zone.
            #    Let's try identifying by label in every frame.
            
            # BETTER STRATEGY: 
            # Identify in center frame.
            # For other frames, pick the pose with bbox center closest to the active_bbox of the adjacent processed frame.
            
            # Let's do a forward pass and backward pass from center.
            
            tracked_active_indices = {} # frame_idx -> pose_idx
            tracked_idle_indices = {}
            
            # Initialize center
            tracked_active_indices[center_idx_in_clip] = active_idx_initial
            tracked_idle_indices[center_idx_in_clip] = idle_idx_initial
            
            # Forward pass
            curr_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            for i in range(center_idx_in_clip + 1, len(frames)):
                poses = poses_per_frame[i]
                if not poses: continue
                
                # Find closest to curr_active_bbox
                best_idx = -1
                min_dist = float('inf')
                
                cx_prev = (curr_active_bbox[0] + curr_active_bbox[2]) / 2
                cy_prev = (curr_active_bbox[1] + curr_active_bbox[3]) / 2
                
                for p_idx, p in enumerate(poses):
                    bbox = unwrap_bbox(p['bbox'])
                    if len(bbox) < 4: continue

                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    dist = (cx - cx_prev)**2 + (cy - cy_prev)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = p_idx
                
                tracked_active_indices[i] = best_idx
                if best_idx != -1:
                    curr_active_bbox = unwrap_bbox(poses[best_idx]['bbox'])
                    # Idle is just the other one (simplification)
                    tracked_idle_indices[i] = get_idle_player(poses, best_idx)
            
            # Backward pass
            curr_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            for i in range(center_idx_in_clip - 1, -1, -1):
                poses = poses_per_frame[i]
                if not poses: continue
                
                best_idx = -1
                min_dist = float('inf')
                
                cx_prev = (curr_active_bbox[0] + curr_active_bbox[2]) / 2
                cy_prev = (curr_active_bbox[1] + curr_active_bbox[3]) / 2
                
                for p_idx, p in enumerate(poses):
                    bbox = unwrap_bbox(p['bbox'])
                    if len(bbox) < 4: continue

                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    dist = (cx - cx_prev)**2 + (cy - cy_prev)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = p_idx

                tracked_active_indices[i] = best_idx
                if best_idx != -1:
                    curr_active_bbox = unwrap_bbox(poses[best_idx]['bbox'])
                    tracked_idle_indices[i] = get_idle_player(poses, best_idx)

            # Generate output
            pose_data_sequence = []
            
            for i in range(len(frames)):
                frame = frames[i]
                poses = poses_per_frame[i]
                
                act_idx = tracked_active_indices.get(i, -1)
                idl_idx = tracked_idle_indices.get(i, -1)
                
                # Overlay
                vis_frame = draw_overlay(frame, poses, act_idx, idl_idx)
                out.write(vis_frame)
                
                # Collect pose data
                if act_idx != -1 and act_idx < len(poses):
                    pose_data_sequence.append(poses[act_idx])
                else:
                    pose_data_sequence.append(None) # Missing data
            
            out.release()
            
            # Save CSV
            pose_csv_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}_pose.csv"
            save_pose_csv(pose_data_sequence, os.path.join(OUTPUT_DIR, pose_csv_filename))
            # print(f"    Saved {clip_filename} and {pose_csv_filename}")

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import parse_shot_csv, get_video_path, identify_player, get_idle_player

# Configuration
SHOTS_CSV_DIR = '/home/daniele/shots_csvs/'
VIDEOS_DIR = '/home/daniele/videos/'
OUTPUT_DIR = 'shot_detector/data/'
MODEL_CONFIG = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
MODEL_CHECKPOINT = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_mmpose():
    if not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Config not found: {MODEL_CONFIG}")
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CHECKPOINT}")
    
    print(f"Initializing MMPose with {MODEL_CONFIG} on {DEVICE}...")
    return MMPoseInferencer(
        pose2d=MODEL_CONFIG,
        pose2d_weights=MODEL_CHECKPOINT,
        device=DEVICE
    )

def extract_clip_and_pose(video_path, start_frame, duration, inferencer):
    """
    Extracts frames, runs pose estimation, and returns data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    poses_per_frame = []
    
    for _ in range(duration):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Run inference on single frame
        # inferencer returns a generator
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        
        # Extract predictions
        # structure: result['predictions'][0][0]['keypoints']... 
        # Actually standard mmpose output format check needed
        # result['predictions'] is a list of results (one per image provided)
        # Each result usually has 'instances' or similar keys.
        # Let's assume standard MMPose output structure
        
        predictions = result['predictions'][0]
        # In newer MMPose, predictions might be an object or dict. 
        # Usually it has 'keypoints', 'bbox', 'bbox_score', 'keypoint_scores'
        
        # We need to normalize this into a list of dicts for our utils
        frame_poses = []
        
        if predictions:
            # prediction is often a list of dicts if multiple people
            # OR a dict with keys containing lists.
            # Let's inspect what MMPoseInferencer returns exactly for RTMO
            # Typically: [{'bbox': [...], 'keypoints': [...], ...}, ...]
            
            # If it's the structure from high-level apis:
            for instance in predictions:
                frame_poses.append(instance)
                
        poses_per_frame.append(frame_poses)

    cap.release()
    return frames, poses_per_frame

def unwrap_bbox(bbox):
    """
    Helper to unwrap nested bbox structure ([x1,y1,x2,y2],) -> [x1,y1,x2,y2]
    """
    if len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
        return bbox[0]
    return bbox

def draw_overlay(frame, poses, active_idx, idle_idx):
    """
    Draws bounding boxes and labels on the frame.
    """
    img = frame.copy()
    
    for i, pose in enumerate(poses):
        bbox = unwrap_bbox(pose['bbox']) # Unwrap here
        if len(bbox) < 4: continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        color = (255, 255, 255) # Default white
        label = ""
        
        if i == active_idx:
            color = (0, 255, 0) # Green for Active
            label = "ACTIVE"
        elif i == idle_idx:
            color = (0, 0, 255) # Red for Idle
            label = "IDLE"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if label:
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        # Draw Keypoints (simplified)
        if 'keypoints' in pose:
            kpts = pose['keypoints']
            for kp in kpts:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0: # valid keypoint
                    cv2.circle(img, (kx, ky), 3, color, -1)

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
    
    # List CSVs
    csv_files = [f for f in os.listdir(SHOTS_CSV_DIR) if f.endswith('.csv')]
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        csv_path = os.path.join(SHOTS_CSV_DIR, csv_file)
        # print(f"Processing {csv_file}...") # tqdm handles this
        
        df = parse_shot_csv(csv_path)
        if df.empty:
            continue
            
        video_path = get_video_path(csv_path, VIDEOS_DIR)
        if not video_path:
            tqdm.write(f"Video not found for {csv_file}")
            continue
            
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
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
            
            # print(f"  > Shot: {shot_type} at frame {center_frame} ({player_label})")
            
            # Extract data
            frames, poses_per_frame = extract_clip_and_pose(video_path, start_frame, duration, inferencer)
            
            if not frames or not poses_per_frame:
                tqdm.write(f"    Failed to extract frames for shot {shot_type} at {center_frame}")
                continue
                
            # Identify players in the CENTER frame (approx index 15)
            # Use the poses from the center frame to decide which index tracks which player
            # Note: Tracking across frames is not guaranteed by simple per-frame inference.
            # However, for 1 second clip, indexes often stay stable if people don't cross.
            # Better approach: Find player in center frame, then find closest match in adjacent frames.
            # For now, let's assume index stability or re-identify per frame?
            # Re-identifying per frame based on spatial consistency is safer.
            
            # Let's pick the active player in the center frame
            center_idx_in_clip = min(15, len(frames) - 1)
            center_poses = poses_per_frame[center_idx_in_clip]
            
            active_idx_initial = identify_player(center_poses, player_label)
            
            if active_idx_initial == -1:
                tqdm.write(f"    Could not find player '{player_label}' in center frame for shot {shot_type} at {center_frame}")
                continue
                
            idle_idx_initial = get_idle_player(center_poses, active_idx_initial)
            
            # Track players across the clip (simple distance tracking)
            active_poses_sequence = []
            
            # Output Video writer
            clip_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}.mp4"
            clip_path = os.path.join(OUTPUT_DIR, clip_filename)
            
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
            
            prev_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            
            # We need to re-process the list to ensure we track the correct person
            # Iterate through all frames
            for i in range(len(frames)):
                frame = frames[i]
                current_poses = poses_per_frame[i]
                
                # Find active player in this frame (closest to previous bbox center)
                # For first frame, we might work backwards from center?
                # Or just assume linear motion.
                # Let's process forward from center, and backward from center.
                pass 

            # SIMPLIFIED TRACKING STRATEGY:
            # 1. Identify active/idle in EVERY frame using the label logic (Left/Right/etc).
            #    Issue: Players move. 'Left' player might become 'Right' if they cross?
            #    But for a shot, they usually stay in their zone.
            #    Let's try identifying by label in every frame.
            
            # BETTER STRATEGY: 
            # Identify in center frame.
            # For other frames, pick the pose with bbox center closest to the active_bbox of the adjacent processed frame.
            
            # Let's do a forward pass and backward pass from center.
            
            tracked_active_indices = {} # frame_idx -> pose_idx
            tracked_idle_indices = {}
            
            # Initialize center
            tracked_active_indices[center_idx_in_clip] = active_idx_initial
            tracked_idle_indices[center_idx_in_clip] = idle_idx_initial
            
            # Forward pass
            curr_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            for i in range(center_idx_in_clip + 1, len(frames)):
                poses = poses_per_frame[i]
                if not poses: continue
                
                # Find closest to curr_active_bbox
                best_idx = -1
                min_dist = float('inf')
                
                cx_prev = (curr_active_bbox[0] + curr_active_bbox[2]) / 2
                cy_prev = (curr_active_bbox[1] + curr_active_bbox[3]) / 2
                
                for p_idx, p in enumerate(poses):
                    bbox = unwrap_bbox(p['bbox'])
                    if len(bbox) < 4: continue

                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    dist = (cx - cx_prev)**2 + (cy - cy_prev)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = p_idx
                
                tracked_active_indices[i] = best_idx
                if best_idx != -1:
                    curr_active_bbox = unwrap_bbox(poses[best_idx]['bbox'])
                    # Idle is just the other one (simplification)
                    tracked_idle_indices[i] = get_idle_player(poses, best_idx)
            
            # Backward pass
            curr_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            for i in range(center_idx_in_clip - 1, -1, -1):
                poses = poses_per_frame[i]
                if not poses: continue
                
                best_idx = -1
                min_dist = float('inf')
                
                cx_prev = (curr_active_bbox[0] + curr_active_bbox[2]) / 2
                cy_prev = (curr_active_bbox[1] + curr_active_bbox[3]) / 2
                
                for p_idx, p in enumerate(poses):
                    bbox = unwrap_bbox(p['bbox'])
                    if len(bbox) < 4: continue

                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    dist = (cx - cx_prev)**2 + (cy - cy_prev)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = p_idx

                tracked_active_indices[i] = best_idx
                if best_idx != -1:
                    curr_active_bbox = unwrap_bbox(poses[best_idx]['bbox'])
                    tracked_idle_indices[i] = get_idle_player(poses, best_idx)

            # Generate output
            pose_data_sequence = []
            
            for i in range(len(frames)):
                frame = frames[i]
                poses = poses_per_frame[i]
                
                act_idx = tracked_active_indices.get(i, -1)
                idl_idx = tracked_idle_indices.get(i, -1)
                
                # Overlay
                vis_frame = draw_overlay(frame, poses, act_idx, idl_idx)
                out.write(vis_frame)
                
                # Collect pose data
                if act_idx != -1 and act_idx < len(poses):
                    pose_data_sequence.append(poses[act_idx])
                else:
                    pose_data_sequence.append(None) # Missing data
            
            out.release()
            
            # Save CSV
            pose_csv_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}_pose.csv"
            save_pose_csv(pose_data_sequence, os.path.join(OUTPUT_DIR, pose_csv_filename))
            # print(f"    Saved {clip_filename} and {pose_csv_filename}")

if __name__ == "__main__":
    main()

