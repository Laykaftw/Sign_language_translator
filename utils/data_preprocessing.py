import os
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

def preprocess_videos(input_dir, output_dir):
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Mask R-CNN on GPU
    model = maskrcnn_resnet50_fpn_v2(pretrained=True).to(device)
    model.eval()
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Process each sign directory
    for sign_dir in os.listdir(input_dir):
        sign_path = os.path.join(input_dir, sign_dir)
        if not os.path.isdir(sign_path):
            continue
        
        # Process each video in the sign directory
        for video_file in os.listdir(sign_path):
            if not video_file.endswith(".mp4"):
                continue
            
            video_path = os.path.join(sign_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            frames_dir = os.path.join(output_dir, sign_dir, video_name, "frames")
            preprocessed_dir = os.path.join(output_dir, sign_dir, video_name, "preprocessed")
            
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Resize frame to 128x128
                    frame = cv2.resize(frame, (128, 128))
                    
                    # Convert to RGB and move to GPU
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = (
                        torch.from_numpy(rgb_frame / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to(device)  # Move tensor to GPU
                    )
                    
                    # Run Mask R-CNN on GPU
                    with torch.no_grad():
                        predictions = model(input_tensor)[0]
                    
                    # Extract hand mask (class 1 is 'person' in COCO; adjust if needed)
                    if "masks" in predictions and len(predictions["masks"]) > 0:
                        best_mask = predictions["masks"][0].cpu().numpy() > 0.5  # Move mask back to CPU
                        segmented_frame = np.zeros_like(rgb_frame)
                        segmented_frame[best_mask[0]] = rgb_frame[best_mask[0]]
                    else:
                        segmented_frame = rgb_frame
                    
                    # Convert to grayscale and normalize
                    gray_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)
                    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
                    normalized_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # Save raw frame
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg"), normalized_frame)
                    
                    # Apply data augmentation and save preprocessed frame
                    augmented_frame = transform(segmented_frame).permute(1, 2, 0).numpy()
                    augmented_frame = (augmented_frame * 127.5 + 127.5).astype(np.uint8)
                    cv2.imwrite(os.path.join(preprocessed_dir, f"frame_{frame_count:04d}.jpg"), augmented_frame)
                    
                    frame_count += 1
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    break
            
            cap.release()
    
    print("Preprocessing complete.")