import cv2
import numpy as np
import torch
from torchvision import transforms, models
from train import CNNSaLSTM, SignLanguageDataset  # Import the dataset and model classes

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset and retrieve class names
dataset = SignLanguageDataset("data/processed", sequence_length=10)
class_names = sorted(dataset.get_class_names())  # Ensure class names are sorted alphabetically
input_size = dataset[0][0].shape[1]
hidden_size = 128
num_classes = len(set(dataset.labels))

# Load the trained model
model = CNNSaLSTM(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load("cnn_sa_lstm_model.pth", weights_only=True))
model.eval()

# Spatial preprocessing (VGG-16 expects RGB images)
spatial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to VGG-16 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Initialize VGG-16 for spatial feature extraction
vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

# Mask R-CNN for background removal
maskrcnn_model = models.detection.maskrcnn_resnet50_fpn_v2(weights=models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device).eval()

def real_time_testing(model, class_names, sequence_length=10):
    cap = cv2.VideoCapture(0)
    buffer = []
    prev_gray_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Remove background using Mask R-CNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = (
            torch.from_numpy(rgb_frame / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        
        with torch.no_grad():
            predictions = maskrcnn_model(input_tensor)[0]

        if "masks" in predictions and len(predictions["masks"]) > 0:
            best_mask = predictions["masks"][0].cpu().numpy() > 0.5
            segmented_frame = np.zeros_like(rgb_frame)
            segmented_frame[best_mask[0]] = rgb_frame[best_mask[0]]
        else:
            segmented_frame = rgb_frame

        # Step 2: Preprocess for spatial features (RGB)
        frame_rgb = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
        spatial_tensor = spatial_transform(frame_rgb).unsqueeze(0).to(device)

        # Extract spatial features using VGG-16
        with torch.no_grad():
            spatial_features = vgg_model(spatial_tensor).cpu().numpy().flatten()

        # Step 3: Preprocess for motion features (Grayscale + Optical Flow)
        gray_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2GRAY)
        gray_frame = cv2.resize(gray_frame, (128, 128))  # Match training motion feature size

        if prev_gray_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_features = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).flatten()
        else:
            motion_features = np.zeros_like(gray_frame).flatten()

        prev_gray_frame = gray_frame

        # Combine spatial and motion features
        combined_features = np.concatenate([spatial_features, motion_features])
        buffer.append(combined_features)

        # Maintain buffer size
        if len(buffer) > sequence_length:
            buffer.pop(0)

        # Predict when buffer is full
        if len(buffer) == sequence_length:
            input_sequence = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_sequence)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                confidence_score = confidence.item() * 100

                # Only display predictions with confidence > 90%
                if confidence_score > 90:
                    predicted_class = class_names[predicted.item()]
                    text = f"Predicted: {predicted_class} ({confidence_score:.1f}%)"
                else:
                    text = "No sign detected"

                # Display results on the frame
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with predictions
        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_testing(model, class_names, sequence_length=10)