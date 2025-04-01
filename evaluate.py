# evaluate.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.dataset import SignLanguageDataset
from models.cnn_sa_lstm import CNNSaLSTM

# Load dataset to determine the number of classes
dataset = SignLanguageDataset("data/processed", sequence_length=10)
num_classes = len(dataset.label_map)  # Dynamically infer the number of classes

# Initialize model with the same parameters used during training
model = CNNSaLSTM(input_size=128*16*16, hidden_size=512, num_classes=num_classes)
model.load_state_dict(torch.load("models/cnn_sa_lstm.pth"))
model.eval()

# DataLoader for evaluation
test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

# Evaluate the model
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.float()
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())

# Print metrics
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}%")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.2f}%")
print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.2f}%")
print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.2f}%")