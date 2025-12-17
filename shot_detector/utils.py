import os
import cv2
import numpy as np
import pandas as pd

def load_fisheye_params(path):
    """
    Load fisheye parameters from a file.
    """
    parameters = {}
    if not os.path.exists(path):
        # Fallback or error
        print(f"Warning: Fisheye params file not found: {path}")
        return None, None

    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split('=')
            if len(parts) == 2:
                key, value = map(str.strip, parts)
                try:
                    parameters[key] = float(value)
                except ValueError:
                    pass

    if not parameters:
        return None, None

    fx = parameters.get('fx', 0)
    fy = parameters.get('fy', 0)
    cx = parameters.get('cx', 0)
    cy = parameters.get('cy', 0)
    k1 = parameters.get('k1', 0)
    k2 = parameters.get('k2', 0)
    p1 = parameters.get('p1', 0)
    p2 = parameters.get('p2', 0)

    mtx = np.array([[fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]])
    dist = np.array([[k1, k2, p1, p2]])

    return mtx, dist

def load_perspective_matrix(path):
    """
    Load perspective transformation matrix from a file.
    """
    if not os.path.exists(path):
        print(f"Warning: Perspective matrix file not found: {path}")
        return None
        
    try:
        matrix = np.loadtxt(path)
        if matrix.shape != (3, 3):
            return None
        return matrix
    except Exception as e:
        print(f"Error loading perspective matrix: {e}")
        return None

def transform_points(points, K=None, D=None, H=None):
    """
    Undistort and/or apply perspective transformation.
    points: List of (x,y) tuples or numpy array (Nx2)
    """
    if points is None or len(points) == 0:
        return np.array([])

    if isinstance(points, list):
        points = np.float32(points)

    # Reshape to (N, 1, 2)
    result = points.reshape(-1, 1, 2)

    if K is not None and D is not None:
        result = cv2.fisheye.undistortPoints(result, K, D, None, K)

    if H is not None:
        result = cv2.perspectiveTransform(result, H)
        
    return result.reshape(-1, 2)

def get_foot_position(bbox):
    """
    Returns the foot position (bottom center) of a bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

# --- Stubs for missing functions required by extract_shots.py ---

def parse_shot_csv(csv_path):
    """
    Parses the shot CSV file.
    Expected columns: Shot, FrameId, Player, etc.
    """
    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        required = ['Shot', 'FrameId', 'Player']
        if not all(col in df.columns for col in required):
            print(f"CSV {csv_path} missing required columns.")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return pd.DataFrame()

def get_video_path(csv_path, videos_dir):
    """
    Infers video path from csv path or content.
    This is a guess implementation.
    """
    # Assuming csv filename matches video filename
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    # Try different extensions
    for ext in ['.mp4', '.mov', '.avi']:
        vid_path = os.path.join(videos_dir, base_name + ext)
        if os.path.exists(vid_path):
            return vid_path
    
    # Fallback: maybe just return constructed path and let caller handle check
    return os.path.join(videos_dir, base_name + ".mp4")

def identify_player(poses, player_label):
    """
    Identifies which pose index corresponds to the player_label.
    This is a tricky one without knowing the logic (e.g. is 'Player' 'Top', 'Bottom', 'Left', 'Right'?).
    
    Placeholder: Returns the index of the first pose.
    """
    if not poses:
        return -1
    # TODO: Implement actual identification logic based on player_label
    # For now, return 0 if exists
    return 0

def get_idle_player(poses, active_idx):
    """
    Returns the index of the player that is NOT the active_idx.
    """
    if not poses or len(poses) < 2:
        return -1
    
    for i in range(len(poses)):
        if i != active_idx:
            return i
    return -1
