import os
import numpy as np
from tqdm import tqdm

def combine_features(video_path, combined_output_dir):
    os.makedirs(combined_output_dir, exist_ok=True)
    
    spatial_dir = os.path.join(video_path, "spatial_features")
    motion_dir = os.path.join(video_path, "motion_features")
    
    spatial_files = sorted([f for f in os.listdir(spatial_dir) if f.startswith("frame")])
    motion_files = sorted([f for f in os.listdir(motion_dir) if f.startswith("motion")])
    
    min_length = min(len(spatial_files), len(motion_files))
    
    for i in tqdm(range(min_length), desc="Combining features"):
        spatial_path = os.path.join(spatial_dir, spatial_files[i])
        motion_path = os.path.join(motion_dir, motion_files[i])
        
        spatial = np.load(spatial_path).flatten()  # Ensure spatial is 1D
        motion = np.load(motion_path).flatten()    # Ensure motion is 1D
        
        combined = np.concatenate([spatial, motion])
        output_path = os.path.join(combined_output_dir, f"combined_{i:04d}.npy")
        np.save(output_path, combined)

if __name__ == "__main__":
    for sign_dir in os.listdir("data/processed"):
        sign_path = os.path.join("data/processed", sign_dir)
        if os.path.isdir(sign_path):
            for video_dir in os.listdir(sign_path):
                video_path = os.path.join(sign_path, video_dir)
                combine_features(video_path, os.path.join(video_path, "combined_features"))