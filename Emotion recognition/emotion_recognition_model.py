# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 07:38:54 2024

@author: Jelena

In this script emotion recognition model is trained.
"""

from mymodel import MyModel
from audiodataset import create_dataloader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
import csv
import os

train_df = pd.read_csv(os.path.join('data','train_dataset.csv'))
val_df = pd.read_csv(os.path.join('data','val_dataset.csv'))

# Set the maximum length of the audio clips (in samples)
train_loader = create_dataloader(train_df, batch_size = 64)
val_loader = create_dataloader(val_df, batch_size = 64)

# init model
num_classes = 5
model = MyModel(num_classes=num_classes)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0006)

# training process
num_epochs = 30

# Initialize the CSV file
csv_file = 'training_log.csv'
csv_headers = ['Epoch', 'Train Loss', 'Val Loss', 'Num Class Train', 'Num Class Val']

# Write headers if file does not exist
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)
        

for epoch in range(0, num_epochs):

    model.train()  # Set the model to training mode
    train_loss = 0.0
    num_batches = len(train_loader)
    all_train_preds = set()

    for batch_idx, (mel_spectrogram, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        # Forward pass
        outputs = model(mel_spectrogram)

        # Calculate the loss
        labels_int = [int(label) for label in labels]
        labels = torch.tensor(labels_int)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Collect predictions
        _, predicted = torch.max(outputs, 1)
        all_train_preds.update(predicted.cpu().numpy())

        # Print progress every few batches
        if batch_idx % 10 == 0:
          print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Training Loss: {train_loss/(batch_idx+1):.4f}')

    train_loss /= num_batches
    num_class_train = len(all_train_preds)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    num_batches = len(val_loader)
    all_val_preds = set()

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (mel_spectrogram, labels) in enumerate(val_loader):
            outputs = model(mel_spectrogram)

            # Calculate the loss
            labels_int = [int(label) for label in labels]
            labels = torch.tensor(labels_int)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Collect predictions
            _, predicted = torch.max(outputs, 1)
            all_val_preds.update(predicted.cpu().numpy())

            # Print progress every few batches
            if batch_idx % 10 == 0:
              print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Validation Loss: {val_loss/(batch_idx+1):.4f}')

    val_loss /= len(val_loader)
    num_class_val = len(all_val_preds)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    # Update the CSV file
    with open(csv_file, mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([epoch + 1, train_loss, val_loss, num_class_train, num_class_val])

    # Save the model
    model_path = f'/content/drive/MyDrive/PhD/Emotion Recognition/model/model_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')        
        
        
        