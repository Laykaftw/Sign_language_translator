# models/cnn_sa_lstm.py
import torch
import torch.nn as nn

class CNNSaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # CNN for spatial analysis (paper uses VGG-16)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Grayscale input (1 channel)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Self-Attention (paper uses multi-head attention)
        self.self_attention = nn.MultiheadAttention(embed_dim=128*16*16, num_heads=8)  # Adjust dimensions
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=128*16*16, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size * sequence_length, channels, height, width)  # Flatten batch and sequence
        x = self.cnn(x)  # CNN feature extraction
        x = x.view(batch_size, sequence_length, -1)  # Reshape to (batch, sequence, features)
        x, _ = self.self_attention(x, x, x)  # Self-Attention
        x, _ = self.lstm(x)  # LSTM
        x = self.fc(x[:, -1, :])  # Use last LSTM output
        return x