import cv2
import time
from tqdm import tqdm
import sys
import os
import torch
import argparse
from mmpose.apis import MMPoseInferencer

def main():
    parser = argparse.ArgumentParser(description='MMPose Video Inference')
    parser.add_argument('video_path', nargs='?', default='data/14-10-BO-0001_short_30s.mp4', help='Path to input video')
    parser.add_argument('output_path', nargs='?', default='results/mmpose/output_mmpose_30s.mp4', help='Path to output video')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint visualization threshold (0.0-1.0)')
    parser.add_argument('--show', action='store_true', help='Show visualization in a window')
    
    args = parser.parse_args()
    
    VIDEO_PATH = args.video_path
    OUTPUT_PATH = args.output_path
    SAVE_OUTPUT = True
    KPT_THR = args.kpt_thr

    print(f"Python: {sys.version}")
    print(f"MMPose: using MMPoseInferencer")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # We use a local config file (downloaded via mim) and local checkpoint.
    # To use smaller models (e.g., rtmo-m, rtmo-s), you need to:
    # 1. Download the corresponding config and checkpoint using mim:
    #    mim download mmpose --config rtmo-s_8xb32-600e_coco-640x640 --dest configs
    # 2. Update the filenames below to match the downloaded files.
    # Available variants: rtmo-s, rtmo-m, rtmo-l
    #rtmo_config = 'configs/rtmo-l_16xb16-600e_coco-640x640.py'
    # rtmo_checkpoint = 'model_weights/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth
    # Using RTMO-s (small)
    rtmo_config = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
    rtmo_checkpoint = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
    
    print("Initializing MMPose RTMO-s (using local config and checkpoint)...")
    try:
        if not os.path.exists(rtmo_config):
             print(f"Config file not found: {rtmo_config}")
             return
        if not os.path.exists(rtmo_checkpoint):
             print(f"Checkpoint file not found: {rtmo_checkpoint}")
             return

        inferencer = MMPoseInferencer(pose2d=rtmo_config, pose2d_weights=rtmo_checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Error initializing inferencer with RTMO config: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {VIDEO_PATH}")
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {fps_video}")
    print(f"Confidence Threshold: {KPT_THR}")

    out = None
    if SAVE_OUTPUT:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))
        print(f"Saving output to {OUTPUT_PATH}")

    pbar = tqdm(total=total_frames, unit="frame", desc="Processing")

    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MMPose inference
        # inferencer can take a single frame (numpy array)
        # return_vis=True returns the visualized image
        try:
            # Pass kpt_thr to control visualization confidence these make better visuals radius=6 thickness=4
            result_generator = inferencer(frame, return_vis=SAVE_OUTPUT, kpt_thr=KPT_THR,  radius=6, thickness=4)
            result = next(result_generator)
            
            if SAVE_OUTPUT and out is not None:
                # result['visualization'] contains the image with keypoints
                # It might be a list of images if batch size > 1, but here it's 1 frame
                if 'visualization' in result and len(result['visualization']) > 0:
                    vis_img = result['visualization'][0]
                    # MMPose might return RGB, OpenCV expects BGR
                    # Visualizer usually outputs BGR if using cv2 backend.
                    # We'll write it directly.
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    out.write(vis_img)
                else:
                    out.write(frame)
        except Exception as e:
            # Fallback if inference fails on a frame
            # print(f"Frame {frame_count} error: {e}")
            if out is not None:
                out.write(frame)
            pass
        
        frame_count += 1
        pbar.update(1)

    cap.release()
    if out is not None:
        out.release()

    end_time = time.time()
    pbar.close()

    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nFinished processing.")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    # Performance Summary for easier reading
    print("\n" + "="*30)
    print(f"PERFORMANCE SUMMARY")
    print("="*30)
    print(f"Method: MMPose RTMO-s")
    print(f"Video: {VIDEO_PATH}")
    print(f"Total Frames: {total_frames}")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Processing Speed: {avg_fps:.2f} FPS")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
