# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import SignLanguageDataset
from models.cnn_sa_lstm import CNNSaLSTM

# Hyperparameters (match the paper)
input_size = 128*16*16  # Output size after CNN + Flatten
hidden_size = 512
num_classes = 26  # Update this based on your dataset
sequence_length = 10
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Dataset and DataLoader
dataset = SignLanguageDataset("data/processed", sequence_length=sequence_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = CNNSaLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.float()
        labels = labels.long()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "models/cnn_sa_lstm.pth")