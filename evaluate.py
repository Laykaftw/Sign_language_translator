import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import SignLanguageDataset, CNNSaLSTM
from sklearn.metrics import classification_report
from tqdm import tqdm

def evaluate_model(model, data_dir, batch_size=32, sequence_length=10):
    # Create dataset and dataloader
    dataset = SignLanguageDataset(data_dir, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate additional metrics (Precision, Recall, F1-Score, etc.)
    report = classification_report(all_labels, all_preds, output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    
    return accuracy, report

if __name__ == "__main__":
    input_directory = "data/processed"
    
    # Load the trained model
    dataset = SignLanguageDataset(input_directory, sequence_length=10)
    input_size = dataset[0][0].shape[1]
    hidden_size = 128
    num_classes = len(set(dataset.labels))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNSaLSTM(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load("cnn_sa_lstm_model.pth"))
    
    # Evaluate the model
    evaluate_model(model, input_directory, sequence_length=10)