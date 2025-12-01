from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm
import sys
import os

# Video path
VIDEO_PATH = 'data/14-10-BO-0001_short.mp4'
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
SAVE_OUTPUT = False
OUTPUT_PATH = 'output_yolopose.mp4'
if len(sys.argv) > 2:
    OUTPUT_PATH = sys.argv[2]
    SAVE_OUTPUT = True

def main():
    print(f"Python: {sys.version}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # Initialize YOLO model
    # 'yolo11n-pose.pt' will be automatically downloaded if not present
    print("Initializing YOLO11n-pose...")
    model_path = "model_weights/yolo11n-pose.pt"
    if not os.path.exists(model_path):
        # Fallback to local if not found, or let YOLO download to current dir then move it?
        # YOLO usually downloads to current dir if name provided. 
        # If path provided, it expects it there.
        # We can just let YOLO download to 'yolo11n-pose.pt' in current dir if we pass the name
        # but the user wants them in model_weights.
        
        # Better strategy: Try to load from model_weights, if fail, download to model_weights?
        # Ultralytics doesn't easily download to custom path unless we handle it.
        # Simplest: Check if exists in model_weights, if so use it. Else use "yolo11n-pose.pt" (will download)
        # and then we could move it. But user wants clean dir.
        
        # Let's assume we download if missing.
        if not os.path.exists("model_weights"):
             os.makedirs("model_weights")
        
        # If we pass just "yolo11n-pose.pt", it downloads to cwd.
        # If we want it in model_weights, we might need to download manually or move it.
        pass

    # We will try to load from model_weights if it exists, otherwise default and warn/move?
    # Actually, if I pass the full path "model_weights/yolo11n-pose.pt" to YOLO(), does it download there?
    # No, usually it errors if path not found, or downloads to cwd if just name.
    
    # Let's try to find it.
    if os.path.exists("model_weights/yolo11n-pose.pt"):
        model = YOLO("model_weights/yolo11n-pose.pt")
    else:
        print("Model not found in model_weights/, downloading...")
        model = YOLO("yolo11n-pose.pt")
        # Move it after download
        import shutil
        if not os.path.exists("model_weights"):
            os.makedirs("model_weights")
        shutil.move("yolo11n-pose.pt", "model_weights/yolo11n-pose.pt")
        # Reload from new location to be safe? Or just proceed (model is loaded in memory)
        print("Moved model to model_weights/")

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
    
    # Process video
    # stream=True is efficient for processing long videos
    # device=0 uses GPU 0, device='cpu' uses CPU
    results_generator = model(VIDEO_PATH, stream=True, verbose=False)
    
    for result in results_generator:
        # If we wanted to visualize:
        if SAVE_OUTPUT and out is not None:
             im_array = result.plot()  # plot a BGR numpy array of predictions
             out.write(im_array)
        
        frame_count += 1
        pbar.update(1)

    end_time = time.time()
    pbar.close()
    
    # Note: Ultralytics 'stream' mode might handle video capture internally if we pass the source path
    # But usually we iterate over the generator.
    # If we wanted to control frame reading manually (like with cv2.VideoCapture), 
    # we would pass the frame image to model() inside a loop.
    # Passing the video file path to model() directly is often more optimized for YOLO.

    if out is not None:
        out.release()
    
    # If we didn't use cv2.VideoCapture to read frames, we might need to rely on what model() processed.
    # However, model(VIDEO_PATH) handles the loop.
    
    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nFinished processing.")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    # Performance Summary for easier reading
    print("\n" + "="*30)
    print(f"PERFORMANCE SUMMARY")
    print("="*30)
    print(f"Method: YOLO11n-pose (Ultralytics)")
    print(f"Video: {VIDEO_PATH}")
    print(f"Total Frames: {total_frames}")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Processing Speed: {avg_fps:.2f} FPS")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()

