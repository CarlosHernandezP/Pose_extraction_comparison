import cv2
import mediapipe as mp
import time
from tqdm import tqdm
import sys
import os

# Video path
# For mediapipe lets stick with this one for now as it is the cropped version.
VIDEO_PATH = '/home/carlos/code/LookAtMeProtoApp/data/intermediate/full_match/period_0/shot_detection/player_1_cropped.mp4'
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
MODEL_PATH = 'model_weights/pose_landmarker_full.task'
SAVE_OUTPUT = False
OUTPUT_PATH = 'output_mediapipe.mp4'
if len(sys.argv) > 2:
    OUTPUT_PATH = sys.argv[2]
    SAVE_OUTPUT = True

def main():
    print(f"Python: {sys.version}")
    print(f"MediaPipe: {mp.__version__}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Imports for the new Tasks API
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    print("Initializing MediaPipe PoseLandmarker (Tasks API) with GPU delegate...")
    
    try:
        # Create options with GPU delegate
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.GPU),
            running_mode=VisionRunningMode.VIDEO
        )
        
        # Initialize Landmarker
        with PoseLandmarker.create_from_options(options) as landmarker:
            
            cap = cv2.VideoCapture(VIDEO_PATH)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Processing video: {VIDEO_PATH}")
            print(f"Total frames: {total_frames}")
            print(f"Video FPS: {fps_video}")

            out = None
            if SAVE_OUTPUT:
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

                # MediaPipe Tasks requires mp.Image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                # Calculate timestamp in milliseconds
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                
                # Process video frame
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if SAVE_OUTPUT and out is not None:
                     # Visualization logic would go here
                     # For now, we just write the original frame as placeholder
                     # or implement drawing if requested later
                     out.write(frame)
                
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
        print(f"Method: MediaPipe (GPU Delegate)")
        print(f"Video: {VIDEO_PATH}")
        print(f"Total Frames: {total_frames}")
        print(f"Processing Time: {total_time:.2f} seconds")
        print(f"Processing Speed: {avg_fps:.2f} FPS")
        print("="*30 + "\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
