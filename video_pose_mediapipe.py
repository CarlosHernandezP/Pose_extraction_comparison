import cv2
import mediapipe as mp
import time
from tqdm import tqdm
import sys
import os
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Video path
VIDEO_PATH = 'data/14-10-BO-0001_short_30s.mp4'
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
MODEL_PATH = 'model_weights/pose_landmarker_full.task'
SAVE_OUTPUT = True
OUTPUT_PATH = 'results/mediapipe/output_30s_upscaled.mp4'
if len(sys.argv) > 2:
    OUTPUT_PATH = sys.argv[2]
    SAVE_OUTPUT = True

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def main():
    print(f"Python: {sys.version}")
    print(f"MediaPipe: {mp.__version__}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: File not found at {VIDEO_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        # Try to find it in root if not found
        if os.path.exists('model_weights/pose_landmarker_full.task'):
             model_path_local = 'model_weights/pose_landmarker_full.task'
        else:
             return
    else:
        model_path_local = MODEL_PATH

    # Imports for the new Tasks API
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Check if input is image or video
    ext = os.path.splitext(VIDEO_PATH)[1].lower()
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']

    running_mode = VisionRunningMode.IMAGE if is_image else VisionRunningMode.VIDEO

    print(f"Initializing MediaPipe PoseLandmarker (Tasks API) with GPU delegate. Mode: {running_mode}")
    
    try:
        # Create options with GPU delegate and Multi-pose support
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path_local, delegate=BaseOptions.Delegate.GPU),
            running_mode=running_mode,
            num_poses=5, # Detect up to 5 people
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize Landmarker
        with PoseLandmarker.create_from_options(options) as landmarker:
            
            if is_image:
                print(f"Processing image: {VIDEO_PATH}")
                frame = cv2.imread(VIDEO_PATH)
                if frame is None:
                    print("Error reading image")
                    return
                
                # MediaPipe Tasks requires mp.Image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                start_time = time.time()
                detection_result = landmarker.detect(mp_image)
                end_time = time.time()
                
                print(f"Detected {len(detection_result.pose_landmarks)} people.")
                
                if SAVE_OUTPUT:
                     annotated_image = draw_landmarks_on_image(image_rgb, detection_result)
                     out_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                     cv2.imwrite(OUTPUT_PATH, out_frame)
                     print(f"Saved output to {OUTPUT_PATH}")
                
                print(f"Processing time: {end_time - start_time:.2f}s")
                
            else:
                # Video Processing
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
                    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else '.', exist_ok=True)
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
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    
                    # Calculate timestamp in milliseconds
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    
                    # Process video frame
                    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    if SAVE_OUTPUT and out is not None:
                         # Visualization
                         annotated_image = draw_landmarks_on_image(image_rgb, detection_result)
                         # Convert back to BGR for OpenCV writing
                         out_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                         out.write(out_frame)
                    
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

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
