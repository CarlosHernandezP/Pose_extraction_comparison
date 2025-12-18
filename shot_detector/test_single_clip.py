#!/usr/bin/env python3
"""
Test script to process a single shot clip for debugging.
Usage:
    python -m shot_detector.test_single_clip <video_path> <frame_number> <player_label> [shot_type]
    
Example:
    python -m shot_detector.test_single_clip videos/22-11-2025-18-10_rpi-LU-0002.mp4 6860 left
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import re
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import identify_player, get_idle_player
from shot_detector.utils import load_fisheye_params, load_perspective_matrix, transform_points, get_foot_position, unwrap_bbox

# Import from extract_shots
from shot_detector.extract_shots import (
    init_mmpose, get_calibration, is_pose_valid, extract_clip_and_pose,
    draw_overlay, save_pose_csv, OUTPUT_DIR,
    match_player_by_position, DISTANCE_THRESHOLD, COURT_DISTANCE_THRESHOLD
)

def test_single_clip(video_path, frame_number, player_label, shot_type="test"):
    """
    Process a single shot clip for testing.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    # Initialize
    print("Initializing MMPose...")
    inferencer = init_mmpose()
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    K, D, H = get_calibration(video_name)
    
    if H is None:
        print(f"Warning: No perspective matrix found for {video_name}")
    
    # Extract clip (15 before, center, 14 after = 30 frames)
    start_frame = max(0, int(frame_number) - 15)
    duration = 30
    
    print(f"Extracting clip: frame {start_frame} to {start_frame + duration - 1} (center: {frame_number})")
    # Get all poses for debugging (to see filtered vs unfiltered)
    frames, poses_per_frame, all_poses_per_frame = extract_clip_and_pose(
        video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=True
    )
    
    if not frames or not poses_per_frame:
        print("Failed to extract frames")
        return
    
    # Identify players in center frame
    center_idx_in_clip = min(15, len(frames) - 1)
    center_poses = poses_per_frame[center_idx_in_clip]
    
    print(f"Found {len(center_poses)} valid poses in center frame")
    
    active_idx_initial = identify_player(center_poses, player_label, K, D, H)
    
    if active_idx_initial == -1:
        print(f"Error: Could not find player '{player_label}' in center frame")
        print("Available poses:", len(center_poses))
        return
    
    idle_idx_initial = get_idle_player(center_poses, active_idx_initial)
    print(f"Active player index: {active_idx_initial}, Idle player index: {idle_idx_initial}")
    
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
    
    # Generate output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    clip_filename = f"{video_name}_{frame_number}_{shot_type}_{player_label}.mp4"
    clip_path = os.path.join(OUTPUT_DIR, clip_filename)
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
    
    pose_data_sequence = []
    last_valid_pose = None
    consecutive_lost_frames = 0
    lost_frames = 0
    MAX_FORWARD_FILL = 5
    
    for i in range(len(frames)):
        frame = frames[i]
        poses = poses_per_frame[i]
        
        act_idx = tracked_active_indices.get(i, -1)
        idl_idx = tracked_idle_indices.get(i, -1)
        
        # Check if we're forward-filling (works for any frame, not just first 5)
        # Forward-fill when: active is lost AND we have a previous valid pose AND we haven't exceeded 5 consecutive lost frames
        is_forward_filled = False
        forward_filled_pose = None
        if act_idx == -1 and last_valid_pose is not None and consecutive_lost_frames < MAX_FORWARD_FILL:
            is_forward_filled = True
            forward_filled_pose = last_valid_pose
            print(f"Frame {i}: Forward-filling active player (consecutive lost frames: {consecutive_lost_frames})")
        
        # Overlay - show all poses for debugging
        all_poses_info = all_poses_per_frame[i] if (all_poses_per_frame and i < len(all_poses_per_frame)) else None
        vis_frame = draw_overlay(frame, poses, act_idx, idl_idx, debug=True, all_poses_info=all_poses_info,
                                is_forward_filled=is_forward_filled, forward_filled_pose=forward_filled_pose)
        out.write(vis_frame)
        
        # Collect pose data with forward-fill (max 5 consecutive frames)
        if act_idx != -1 and act_idx < len(poses):
            last_valid_pose = poses[act_idx]
            pose_data_sequence.append(poses[act_idx])
            consecutive_lost_frames = 0  # Reset counter
        else:
            lost_frames += 1
            # Use previous frame's pose if available and within limit
            if last_valid_pose is not None and consecutive_lost_frames < MAX_FORWARD_FILL:
                pose_data_sequence.append(last_valid_pose)
                consecutive_lost_frames += 1
            else:
                pose_data_sequence.append(None)  # Lost for too long or never had valid pose
                consecutive_lost_frames += 1
    
    out.release()
    
    # Save CSV
    pose_csv_filename = f"{video_name}_{frame_number}_{shot_type}_{player_label}_pose.csv"
    save_pose_csv(pose_data_sequence, os.path.join(OUTPUT_DIR, pose_csv_filename))
    
    print(f"\nDone!")
    print(f"Output video: {clip_path}")
    print(f"Output CSV: {os.path.join(OUTPUT_DIR, pose_csv_filename)}")
    print(f"Lost frames (forward-filled): {lost_frames}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    
    video_path = sys.argv[1]
    frame_number = int(sys.argv[2])
    player_label = sys.argv[3]
    shot_type = sys.argv[4] if len(sys.argv) > 4 else "test"
    
    test_single_clip(video_path, frame_number, player_label, shot_type)

