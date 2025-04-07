import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define a custom dataset
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []
        self.label_map = {}
        self.class_names = []  # List to store class names
        
        # Sort sign directories to ensure consistent label order
        for idx, sign_dir in enumerate(sorted(os.listdir(data_dir))):  # Added sorted()
            sign_path = os.path.join(data_dir, sign_dir)
            if not os.path.isdir(sign_path):
                continue
            self.label_map[sign_dir] = idx
            self.class_names.append(sign_dir)  # Store class names in order
            
            for video_dir in os.listdir(sign_path):
                spatial_dir = os.path.join(sign_path, video_dir, "spatial_features")
                motion_dir = os.path.join(sign_path, video_dir, "motion_features")
                
                if not os.path.exists(spatial_dir) or not os.path.exists(motion_dir):
                    print(f"Skipping incomplete video: {video_dir}")
                    continue
                
                spatial_files = sorted(os.listdir(spatial_dir))
                motion_files = sorted(os.listdir(motion_dir))
                min_length = min(len(spatial_files), len(motion_files))
                
                for i in range(min_length - sequence_length + 1):
                    self.data.append((spatial_dir, motion_dir, i))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spatial_dir, motion_dir, start_idx = self.data[idx]
        spatial_files = sorted(os.listdir(spatial_dir))
        motion_files = sorted(os.listdir(motion_dir))
        
        spatial_seq = []
        motion_seq = []
        for j in range(start_idx, start_idx + self.sequence_length):
            s = np.load(os.path.join(spatial_dir, spatial_files[j])).flatten()
            m = np.load(os.path.join(motion_dir, motion_files[j])).flatten()
            spatial_seq.append(s)
            motion_seq.append(m)
        
        combined_seq = np.concatenate([spatial_seq, motion_seq], axis=1)
        return torch.tensor(combined_seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
    # NEW: Add this method to retrieve class names
    def get_class_names(self):
        return self.class_names

class CNNSaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNSaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # Ensure batch_first=True
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Input shape: (batch_size, sequence_length, input_size)
        out = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return out

# Training loop
def train_model(data_dir, epochs=10, batch_size=32, learning_rate=0.001, sequence_length=10):
    # Create dataset and dataloader
    dataset = SignLanguageDataset(data_dir, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    input_size = dataset[0][0].shape[1]  # Infer input size from the first sample
    hidden_size = 128
    num_classes = len(set(dataset.labels))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNSaLSTM(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "cnn_sa_lstm_model.pth")
    print("Model saved as cnn_sa_lstm_model.pth")

if __name__ == "__main__":
    input_directory = "data/processed"
    train_model(input_directory, sequence_length=10)  # Set sequence length to 10