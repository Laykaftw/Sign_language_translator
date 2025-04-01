import os
import numpy as np
import torch    
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Load VGG-16 for feature extraction
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg16 = torch.nn.Sequential(
    *list(vgg16.children())[:-1],  # Remove final classification layer
    torch.nn.AdaptiveAvgPool2d((1, 1))  # Add global pooling to flatten spatial dimensions
)
vgg16.eval()

# Preprocessing for VGG-16
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_spatial_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    if not frame_files:
        print(f"No frames found in {input_dir}")
        return
    
    for frame_file in tqdm(frame_files, desc="Extracting spatial features"):
        frame_path = os.path.join(input_dir, frame_file)
        img = Image.open(frame_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            features = vgg16(img_tensor).squeeze().flatten().numpy()  # Flatten to 1D
        
        output_path = os.path.join(output_dir, f"{os.path.splitext(frame_file)[0]}.npy")
        np.save(output_path, features)
        print(f"Saved spatial features: {output_path}")

if __name__ == "__main__":
    for sign_dir in os.listdir("data/processed"):
        sign_path = os.path.join("data/processed", sign_dir)
        if os.path.isdir(sign_path):
            for video_dir in os.listdir(sign_path):
                video_path = os.path.join(sign_path, video_dir)
                preprocessed_dir = os.path.join(video_path, "preprocessed")
                spatial_output_dir = os.path.join(video_path, "spatial_features")
                
                if os.path.exists(preprocessed_dir):
                    extract_spatial_features(preprocessed_dir, spatial_output_dir)
                else:
                    print(f"Skipping {video_path}: 'preprocessed' folder missing")