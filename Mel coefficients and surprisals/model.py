# -*- coding: utf-8 -*-
"""model.ipynb

@author: Jelena

Ovdje se vrsi treniranje modela za predikciju spektra na osnovu parametara kao
sto su tokeni teksta i vrijednosti surprisala.

"""
import pandas as pd
import os

# Load DataFrame back from pickle file
pkl_path = os.path.join('..','podaci','general_data.pkl')
df = pd.read_pickle(pkl_path)

df.info()

batch_size = 256
vocab_size = 276

# Mapping dictionary: map 10 folds to 5 folds
fold_mapping = {
    0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4
}

# Apply the mapping to the Folds column
df['Fold'] = df['Fold'].map(fold_mapping)

k_test = 4  
k_val = 0  # k_test + 1
train_data = df[df['Fold'] != k_test]
train_data = train_data[train_data['Fold'] != k_val]
test_data = df[df['Fold'] == k_test]
val_data = df[df['Fold'] == k_val]

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file, _, text, surprisal_values, speaker, emotion, _, _, mel_targets = self.data.iloc[idx]
        return torch.tensor(text), torch.tensor(surprisal_values), torch.tensor(mel_targets), idx
    
# Instantiate your dataset
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
val_dataset = CustomDataset(val_data)

# Create a data loader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)    


import torch.nn as nn
import torch.nn.functional as F

class TacotronWithSurprisal(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, mel_dim, batch_size):
        super(TacotronWithSurprisal, self).__init__()

        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Surprisal embedding layer
        self.surprisal_embedding = nn.Linear(1, embedding_dim)

        # Encoder
        self.encoder = nn.LSTM(embedding_dim, encoder_hidden_dim, batch_first=True)

        # Attention mechanism
        self.attention = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, 1)

        # Decoder
        self.decoder = nn.LSTMCell(embedding_dim + encoder_hidden_dim, decoder_hidden_dim)

        # Post processing
        #self.linear = nn.Linear(decoder_hidden_dim, 250)
        self.fc1 = nn.Linear(13*256, 1024)  # (input_dimension, hidden_dim)
        self.fc2 = nn.Linear(1024, 80*250)  # (hidden_dimension, output_dimension)

        # Learnable initial decoder input
        self.init_decoder_input = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, text, surprisal_values):
        # Embed text
        text_embedded = self.embedding(text)
        # Embed surprisal values
        surprisal_embedded = self.surprisal_embedding(surprisal_values.unsqueeze(-1))
        # Combine text and surprisal embeddings
        combined_embedded = text_embedded + surprisal_embedded

        # Encode the combined embeddings
        encoder_outputs, (h_n, c_n) = self.encoder(combined_embedded)

        # Initialize the decoder
        batch_size = text.size(0)
        seq_len = text.size(1)
        decoder_hidden = (h_n[-1], c_n[-1])
        decoder_input = self.init_decoder_input.expand(batch_size, -1)  # Initialize with learned parameter

        # Prepare for attention mechanism
        decoder_outputs = []
        for t in range(seq_len):
            # Compute attention weights
            decoder_hidden_broadcasted = decoder_hidden[0].unsqueeze(1).expand(batch_size, seq_len, decoder_hidden[0].size(1))
            attn_input = torch.cat((decoder_hidden_broadcasted, encoder_outputs), dim=2)
            attn_weights = F.softmax(self.attention(attn_input), dim=1)
            context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)

            # Decoder step
            decoder_input_combined = torch.cat((decoder_input, context_vector), dim=-1)
            decoder_hidden = self.decoder(decoder_input_combined, decoder_hidden)
            decoder_outputs.append(decoder_hidden[0])

            # Update decoder input for the next time step
            decoder_input = self.init_decoder_input.expand(batch_size, -1)

        # Post processing
        # Stack sequences along a new axis
        stacked_outputs = torch.stack(decoder_outputs, dim=1)
        # Flatten the last two dimensions (13 and 256) into one
        flattened_outputs = stacked_outputs.view(batch_size, -1)  # Shape: (batch_size, 13*256)
        x = self.fc1(flattened_outputs)
        x = torch.relu(x)
        transformed_outputs = self.fc2(x)

        # Final mel outputs
        mel_outputs = transformed_outputs.view(batch_size, 80, 250)

        return mel_outputs
    
    
# Model instantiation example
embedding_dim = 128
encoder_hidden_dim = 256
decoder_hidden_dim = 256
mel_dim = 80

model = TacotronWithSurprisal(vocab_size, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, mel_dim, batch_size)

# Example input data
text = torch.randint(0, vocab_size, (batch_size, 13))  # Batch size 57, sequence length 13
surprisal_values = torch.randn(batch_size, 13)

# Forward pass
mel_outputs = model(text, surprisal_values)
print(mel_outputs.shape)  # Should be (batch_size, sequence_length, target_dim)


from tqdm import tqdm  # Import tqdm for the progress bar

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# Directory to save the best model
save_dir = './best_model'
os.makedirs(save_dir, exist_ok=True)
best_val_loss = float('inf')

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Example training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # Use tqdm to create a progress bar for the training phase
    with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch in train_dataloader:
            text, surprisal_values, mel_targets, _ = batch
            optimizer.zero_grad()

            # Forward pass
            mel_outputs = model(text, surprisal_values)
            # Compute loss
            loss = criterion(mel_outputs, mel_targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update the progress bar
            pbar.set_postfix({'train_loss': train_loss / (pbar.n + 1)})
            pbar.update(1)

    # Compute average training loss
    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)  # Store the training loss

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            text, surprisal_values, mel_targets,_ = batch

            # Forward pass
            mel_outputs = model(text, surprisal_values)
            # Compute loss
            loss = criterion(mel_outputs, mel_targets)
            val_loss += loss.item()

    # Compute average validation loss
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)  # Store the validation loss

    print(f"Epoch {epoch+1}/{num_epochs},\n Training Loss: {train_loss}, Validation Loss: {val_loss}")

    # Save the model if the validation loss is the best we've seen so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(save_dir, f'best_model_{k_test}.pth')
        torch.save(model.state_dict(), best_model_path)
        print('Model saved!')



# Load the best model
best_model_path = os.path.join(save_dir, f'best_model_{k_test}.pth')
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Lists to store the test results
audio_files = []
speakers = []
emotions = []
model_outputs = []
test_losses = []

with torch.no_grad():
    for batch in test_dataloader:
        text, surprisal_values, mel_targets, indices = batch

        # Forward pass
        mel_outputs = model(text, surprisal_values)

        # Compute loss for each example in the batch
        for i in range(text.size(0)):
            single_mel_output = mel_outputs[i].unsqueeze(0)
            single_mel_target = mel_targets[i].unsqueeze(0)
            loss = criterion(single_mel_output, single_mel_target)

            # Collect the results
            model_outputs.append(single_mel_output.cpu().numpy())
            test_losses.append(loss.item())

        # Collect corresponding data info
        indices = indices.numpy()  # Convert indices to numpy array
        audio_files.extend(test_data.iloc[indices]['Audio Files'].tolist())
        speakers.extend(test_data.iloc[indices]['Speaker'].tolist())
        emotions.extend(test_data.iloc[indices]['Emotion'].tolist())

# Compute average test loss
test_loss = sum(test_losses) / len(test_losses)
print(f"Test Loss (with best model): {test_loss}")

# Combine the results into a DataFrame
results_df = pd.DataFrame({
    'Audio Files': audio_files,
    'Speaker': speakers,
    'Emotion': emotions,
    'Test Loss': test_losses
})

# Save results to a CSV file
results_csv_path = os.path.join('..','podaci','mel results', f'results_surprisal_{k_test}.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Test results saved to {results_csv_path}")



import matplotlib.pyplot as plt

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()










