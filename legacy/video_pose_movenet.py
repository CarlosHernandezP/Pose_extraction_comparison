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

# MoveNet MultiPose constants
# MultiPose Lightning accepts dynamic input, but resizing to multiple of 32 (e.g. 256) is recommended
# Using a larger size (e.g. 352, 384, 512) helps with detecting small people in wide shots
INPUT_SIZE = 384 
MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"
MAX_PEOPLE = 2 # Limit to the top 2 most confident predictions

def draw_keypoints(frame, keypoints_list, confidence_threshold=0.1):
    """
    Draws keypoints for multiple people with aspect ratio correction.
    keypoints_list: [N, 17, 3] array where last dim is (y, x, score)
    """
    height, width, _ = frame.shape
    max_dim = max(height, width)
    
    # Calculate padding offsets in original image space
    # The model "sees" the image centered in a square of size max_dim
    pad_y_offset = (max_dim - height) / 2
    pad_x_offset = (max_dim - width) / 2

    # Edges for MoveNet (COCO format)
    EDGES = {
        (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', 
        (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm', 
        (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm', 
        (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', 
        (12, 14): 'c', (14, 16): 'c'
    }
    
    # Iterate over each person
    # keypoints_list is sorted by confidence score, so taking the top MAX_PEOPLE works for filtering
    for person_kps in keypoints_list[:MAX_PEOPLE]:
        # person_kps is [17, 3] -> (y, x, score)
        
        scaled_kps = []
        for kp in person_kps:
            ky, kx, k_conf = kp
            # Convert normalized coordinates back to original image pixels
            # taking into account the padding added during resize_with_pad
            pixel_y = ky * max_dim - pad_y_offset
            pixel_x = kx * max_dim - pad_x_offset
            
            scaled_kps.append([pixel_y, pixel_x, k_conf])
        
        scaled_kps = np.array(scaled_kps)
        
        # Check if enough keypoints are detected with high confidence to draw the person
        # e.g. at least 3 keypoints > threshold
        valid_kps = np.sum(scaled_kps[:, 2] > confidence_threshold)
        if valid_kps < 2:
            continue

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

def process_frame(frame, movenet):
    # Prepare input
    # MultiPose expects [1, height, width, 3] int32 tensor.
    # Resize to multiple of 32 (e.g. 256) with padding
    # IMPORTANT: MoveNet expects RGB, but OpenCV gives BGR
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img_expanded = np.expand_dims(img_rgb, axis=0)
    resized_image = tf.image.resize_with_pad(img_expanded, INPUT_SIZE, INPUT_SIZE)
    input_image = tf.cast(resized_image, dtype=tf.int32)

    # Inference
    outputs = movenet(input_image)
    # Output shape: [1, 6, 56]
    # Each person (up to 6) has 56 values:
    # 17 * 3 (y, x, score) = 51 keypoint values
    # + 5 (ymin, xmin, ymax, xmax, score) bounding box values
    raw_output = outputs['output_0'].numpy()
    
    # Reshape to [6, 17, 3] for easier drawing
    # We only take the first 51 values for keypoints
    # The output is sorted by score, so first person is most confident
    keypoints_all_people = raw_output[0, :, :51].reshape((6, 17, 3))
    
    return keypoints_all_people

def main():
    print(f"Python: {sys.version}")
    print(f"TensorFlow Version: {tf.__version__}")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: File not found at {VIDEO_PATH}")
        return

    print(f"Loading MoveNet MultiPose from: {MODEL_URL}")
    print(f"Model cache dir: {os.environ.get('TFHUB_CACHE_DIR')}")
    print(f"Max people to draw: {MAX_PEOPLE}")
    model = hub.load(MODEL_URL)
    movenet = model.signatures['serving_default']
    print("Model loaded.")

    # Check if input is image or video
    # Simple check by extension
    ext = os.path.splitext(VIDEO_PATH)[1].lower()
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']

    if is_image:
        print(f"Processing image: {VIDEO_PATH}")
        frame = cv2.imread(VIDEO_PATH)
        if frame is None:
            print("Error reading image")
            return
            
        start_time = time.time()
        keypoints_all_people = process_frame(frame, movenet)
        
        if SAVE_OUTPUT:
            draw_keypoints(frame, keypoints_all_people)
            # Ensure output path has image extension
            out_path = OUTPUT_PATH
            out_ext = os.path.splitext(out_path)[1].lower()
            if out_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                out_path = os.path.splitext(out_path)[0] + ".jpg"
                
            cv2.imwrite(out_path, frame)
            print(f"Saved output to {out_path}")
            
        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f}s")
        
    else:
        # Video processing
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
                
            keypoints_all_people = process_frame(frame, movenet)

            if SAVE_OUTPUT and out is not None:
                draw_keypoints(frame, keypoints_all_people)
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
        print(f"Method: MoveNet MultiPose Lightning (TensorFlow)")
        print(f"Video: {VIDEO_PATH}")
        print(f"Total Frames: {total_frames}")
        print(f"Processing Time: {total_time:.2f} seconds")
        print(f"Processing Speed: {avg_fps:.2f} FPS")
        print("="*30 + "\n")

if __name__ == "__main__":
    main()
