import os
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
import torch
from PIL import Image

def extract_spatial_features(input_dir, output_dir):
    """
    Extract spatial features from preprocessed frames using the VGG-16 model.
    """
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the VGG-16 model (pretrained on ImageNet)
    model = models.vgg16(pretrained=True).features.to(device)
    model.eval()
    
    # Define preprocessing pipeline for VGG-16
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VGG-16 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
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
            
            # Create output directory for spatial features
            output_video_dir = os.path.join(output_dir, sign_dir, video_dir, "spatial_features")
            os.makedirs(output_video_dir, exist_ok=True)
            
            # Process each preprocessed frame
            for frame_file in tqdm(os.listdir(video_path), desc=f"Processing {sign_dir}/{video_dir}"):
                frame_path = os.path.join(video_path, frame_file)
                try:
                    # Load and preprocess the frame
                    frame = Image.open(frame_path).convert("RGB")
                    input_tensor = transform(frame).unsqueeze(0).to(device)  # Add batch dimension
                    
                    # Extract spatial features using VGG-16
                    with torch.no_grad():
                        features = model(input_tensor).cpu().numpy()  # Move features back to CPU
                    
                    # Save the extracted features as a .npy file
                    output_path = os.path.join(output_video_dir, os.path.splitext(frame_file)[0] + ".npy")
                    np.save(output_path, features)
                except Exception as e:
                    print(f"Error processing frame {frame_file}: {e}")
    
    print("Spatial feature extraction complete.")

if __name__ == "__main__":
    # Define input and output directories
    input_directory = "data/processed"
    output_directory = "data/processed"
    
    # Run spatial feature extraction
    extract_spatial_features(input_directory, output_directory)