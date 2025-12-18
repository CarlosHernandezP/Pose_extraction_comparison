import os
import cv2
import numpy as np
import argparse
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import load_fisheye_params, load_perspective_matrix, transform_points, get_foot_position, unwrap_bbox

# Configuration
PARAM_DIR = 'parameters'
FISHEYE_FILE = 'fishcam-fisheye.txt'

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

def draw_court_debug(img, poses, K, D, H):
    """
    Draws a top-down view of the court with player positions.
    """
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
        
        foot_pos = get_foot_position(bbox)
        
        # Transform
        if H is not None:
            transformed = transform_points([foot_pos], K, D, H)
            if len(transformed) > 0:
                tx, ty = transformed[0]
                
                # Check valid (matching extract_shots.py bounds)
                is_in_court = (-0.3 <= tx <= 10.3 and -1.3 <= ty <= 21.3)
                is_low_enough = (foot_pos[1] > 0.25 * h) # Note: 0.25 threshold
                
                color = (0, 255, 0) if (is_in_court and is_low_enough) else (0, 0, 255)
                
                # Draw on Debug Map
                dpx, dpy = to_debug(tx, ty)
                cv2.circle(debug_img, (dpx, dpy), 8, color, -1)
                cv2.putText(debug_img, f"P{i}", (dpx+10, dpy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Text Info
                info_y = 30 + i * 30
                status = "OK" if (is_in_court and is_low_enough) else "FAIL"
                reason = []
                if not is_in_court: reason.append(f"Out: {tx:.1f},{ty:.1f}")
                if not is_low_enough: reason.append(f"High: y={foot_pos[1]}/{h}")
                
                cv2.putText(debug_img, f"P{i}: {status} {' '.join(reason)}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw on Main Image
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"P{i} ({tx:.1f}, {ty:.1f})", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
    return np.hstack((img, debug_img))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Start frame')
    parser.add_argument('--duration', type=int, default=100, help='Duration frames')
    args = parser.parse_args()
    
    video_name = os.path.basename(args.video_path)
    K, D, H = get_calibration(video_name)
    
    # Init model
    model_config = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
    model_ckpt = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
    inferencer = MMPoseInferencer(pose2d=model_config, pose2d_weights=model_ckpt)
    
    cap = cv2.VideoCapture(args.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    
    out_path = f"debug_{video_name}_{args.frame}.mp4"
    print(f"Writing debug video to {out_path}")
    
    vw = None
    
    for _ in range(args.duration):
        ret, frame = cap.read()
        if not ret: break
        
        # Inference
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        predictions = result['predictions'][0]
        
        # Visualize
        vis = draw_court_debug(frame, predictions, K, D, H)
        
        if vw is None:
            h, w = vis.shape[:2]
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            
        vw.write(vis)
        
    cap.release()
    if vw: vw.release()
    print("Done.")

if __name__ == "__main__":
    main()

