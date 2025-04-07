import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_motion_features(input_dir, output_dir):
    """
    Extract motion features from preprocessed frames using the Lucas-Kanade optical flow method.
    """
    # Process each sign directory
    for sign_dir in os.listdir(input_dir):
        sign_path = os.path.join(input_dir, sign_dir)
        if not os.path.isdir(sign_path):
            continue
        
        # Process each video in the sign directory
        for video_dir in os.listdir(sign_path):
            video_path = os.path.join(sign_path, video_dir, "preprocessed")
            if not os.path.exists(video_path):
                continue
            
            # Create output directory for motion features
            output_video_dir = os.path.join(output_dir, sign_dir, video_dir, "motion_features")
            os.makedirs(output_video_dir, exist_ok=True)
            
            # Load all preprocessed frames
            frame_files = sorted(os.listdir(video_path))
            prev_frame = None
            
            for i, frame_file in enumerate(tqdm(frame_files, desc=f"Processing {sign_dir}/{video_dir}")):
                frame_path = os.path.join(video_path, frame_file)
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                
                if prev_frame is not None:
                    # Compute optical flow using Lucas-Kanade method
                    flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    
                    # Extract magnitude and angle of flow vectors
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    
                    # Save motion features as a .npy file
                    output_path = os.path.join(output_video_dir, os.path.splitext(frame_file)[0] + ".npy")
                    np.save(output_path, magnitude)
                
                prev_frame = frame

if __name__ == "__main__":
    # Define input and output directories
    input_directory = "data/processed"
    output_directory = "data/processed"
    
    # Run motion feature extraction
    extract_motion_features(input_directory, output_directory)