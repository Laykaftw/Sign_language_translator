import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_motion_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    prev_frame = None
    
    for i, frame_file in enumerate(tqdm(frame_files, desc="Extracting motion features")):
        frame_path = os.path.join(input_dir, frame_file)
        current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_frame is not None:
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Resize and flatten
            flow_resized = cv2.resize(flow, (64, 64))
            flow_flattened = flow_resized.flatten()
            
            # Save motion features
            output_path = os.path.join(output_dir, f"motion_{i:04d}.npy")
            np.save(output_path, flow_flattened)
        
        prev_frame = current_frame

if __name__ == "__main__":
    for sign_dir in os.listdir("data/processed"):
        sign_path = os.path.join("data/processed", sign_dir)
        if os.path.isdir(sign_path):
            for video_dir in os.listdir(sign_path):
                video_path = os.path.join(sign_path, video_dir)
                preprocessed_dir = os.path.join(video_path, "preprocessed")
                motion_output_dir = os.path.join(video_path, "motion_features")
                
                if os.path.exists(preprocessed_dir):
                    extract_motion_features(preprocessed_dir, motion_output_dir)