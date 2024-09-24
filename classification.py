import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClapProcessor, ClapModel
from sklearn.metrics import accuracy_score
from audio_datasets.esc50_dataset import get_esc50_data_loaders

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    sys.path.append('/content/py/isp_final_project')  # For Colab

# Define the classifier model
class CLAPClassifier(nn.Module):
    def __init__(self, clap_model, num_classes):
        super(CLAPClassifier, self).__init__()
        self.clap_model = clap_model
        self.fc = nn.Linear(512, num_classes)

    def forward(self,audio_features):
        with torch.no_grad():
            outputs = self.clap_model(
                is_longer=audio_features['is_longer'].to(DEVICE),
                input_ids=audio_features['input_ids'].squeeze(1).to(DEVICE),  # Text input
                input_features=audio_features['input_features'].squeeze(1).to(DEVICE),  # Audio input
                attention_mask=audio_features['attention_mask'].to(DEVICE))
        audio_embeds = outputs['audio_embeds']
        logits = self.fc(audio_embeds)
        return logits

def evaluate_predictions(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for param in model.clap_model.parameters():
        param.requires_grad = False
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            audio_features, true_labels = batch

            # Forward pass
            optimizer.zero_grad()
            logits = model(audio_features)

            # Calculate loss
            loss = criterion(logits, true_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            audio_features, true_labels = batch

            # Forward pass
            logits = model(audio_features)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())

    evaluate_predictions(all_predictions, all_true_labels)

# Initialize the processor and model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused")
clap_model.to(DEVICE)

# Load data
train_loader, test_loader = get_esc50_data_loaders(False)

# Get the number of classes from the dataset
captions = train_loader.dataset.dataset.categories_id
num_classes = len(captions)

# Initialize the classifier
model = CLAPClassifier(clap_model, num_classes)
model.to(DEVICE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# Train the model
num_epochs = 10  # Adjust as necessary
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model, test_loader)
