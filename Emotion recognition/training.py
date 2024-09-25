# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:07:55 2024

@author: comsol
"""

from mymodel import MyModel
from audiodataset import create_dataloader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
import csv
import os

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_df = pd.read_csv(os.path.join('data', 'train_dataset.csv'))
val_df = pd.read_csv(os.path.join('data', 'val_dataset.csv'))

# Set the maximum length of the audio clips (in samples)
train_loader = create_dataloader(train_df, batch_size=64)
val_loader = create_dataloader(val_df, batch_size=64)

# Init model and move to GPU
num_classes = 5
model = MyModel(num_classes=num_classes).to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0006)

# Training process
num_epochs = 30

# Initialize the CSV file
csv_file = 'training_log.csv'
csv_headers = ['Epoch', 'Train Loss', 'Val Loss', 'Num Class Train', 'Num Class Val']

# Write headers if file does not exist
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    num_batches = len(train_loader)
    all_train_preds = set()

    for batch_idx, (mel_spectrogram, labels) in enumerate(train_loader):
        # Move data to GPU
        mel_spectrogram = mel_spectrogram.to(device)
        labels_int = [int(label) for label in labels]
        labels = torch.tensor(labels_int).to(device)  # Move labels to GPU

        optimizer.zero_grad()
        # Forward pass
        outputs = model(mel_spectrogram)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Collect predictions
        _, predicted = torch.max(outputs, 1)
        all_train_preds.update(predicted.cpu().numpy())

    train_loss /= num_batches
    num_class_train = len(all_train_preds)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    num_batches = len(val_loader)
    all_val_preds = set()

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (mel_spectrogram, labels) in enumerate(val_loader):
            # Move validation data to GPU
            mel_spectrogram = mel_spectrogram.to(device)
            labels_int = [int(label) for label in labels]
            labels = torch.tensor(labels_int).to(device)  # Move labels to GPU

            # Forward pass
            outputs = model(mel_spectrogram)

            # Calculate the loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Collect predictions
            _, predicted = torch.max(outputs, 1)
            all_val_preds.update(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    num_class_val = len(all_val_preds)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    # Update the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_loss, num_class_train, num_class_val])

    # Save the model
    model_path = f'model/model_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

