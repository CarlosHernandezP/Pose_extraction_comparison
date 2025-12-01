import os
import sys

# Set TFHUB_CACHE_DIR before importing tensorflow_hub to ensure it's picked up
# We want to save models in the local 'model_weights' folder
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.getcwd(), "model_weights")
from tqdm import tqdm
import time
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Video path
VIDEO_PATH = 'data/14-10-BO-0001_short.mp4'
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
SAVE_OUTPUT = True
OUTPUT_PATH = 'output_movenet.mp4'
if len(sys.argv) > 2:
    OUTPUT_PATH = sys.argv[2]
    SAVE_OUTPUT = True

# MoveNet SinglePose Thunder constants
# Thunder expects 256x256 input
INPUT_SIZE = 256 
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    """
    Draws keypoints for a single person.
    keypoints: [17, 3] array where last dim is (y, x, score)
    """
    y_ratio, x_ratio = frame.shape[0], frame.shape[1]
    
    # Edges for MoveNet (COCO format)
    EDGES = {
        (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', 
        (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm', 
        (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm', 
        (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', 
        (12, 14): 'c', (14, 16): 'c'
    }
    
    # Scale coordinates
    # shape is (y, x, score)
    # y, x are normalized [0, 1]
    
    scaled_kps = []
    for kp in keypoints:
        ky, kx, k_conf = kp
        scaled_kps.append([ky * y_ratio, kx * x_ratio, k_conf])
    scaled_kps = np.array(scaled_kps)
    
    # Draw edges
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = scaled_kps[p1]
        y2, x2, c2 = scaled_kps[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Draw keypoints
    for kp in scaled_kps:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)

def main():
    print(f"Python: {sys.version}")
    print(f"TensorFlow Version: {tf.__version__}")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    print(f"Loading MoveNet SinglePose Thunder from: {MODEL_URL}")
    print(f"Model cache dir: {os.environ.get('TFHUB_CACHE_DIR')}")
    model = hub.load(MODEL_URL)
    movenet = model.signatures['serving_default']
    print("Model loaded.")

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
            
        # Prepare input
        # SinglePose Thunder expects [1, 256, 256, 3] int32 tensor.
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), INPUT_SIZE, INPUT_SIZE)
        input_image = tf.cast(img, dtype=tf.int32)

        # Inference
        outputs = movenet(input_image)
        # Output shape: [1, 1, 17, 3]
        raw_output = outputs['output_0'].numpy()
        
        # Extract keypoints: [17, 3] -> (y, x, score)
        keypoints = raw_output[0, 0, :, :]

        if SAVE_OUTPUT and out is not None:
            draw_keypoints(frame, keypoints)
            out.write(frame)
        
        frame_count += 1
        pbar.update(1)

    end_time = time.time()
    pbar.close()
    cap.release()
    
    if out is not None:
        out.release()
    
    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nFinished processing.")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    # Performance Summary
    print("\n" + "="*30)
    print(f"PERFORMANCE SUMMARY")
    print("="*30)
    print(f"Method: MoveNet SinglePose Thunder (TensorFlow)")
    print(f"Video: {VIDEO_PATH}")
    print(f"Total Frames: {total_frames}")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Processing Speed: {avg_fps:.2f} FPS")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
