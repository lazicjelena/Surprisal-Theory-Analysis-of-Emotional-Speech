# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:18:26 2024

@author: Jelena
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torch


mel_dim = 80

def get_fixed_length_mel_spectrogram(y, sr, n_mels, fixed_length):
    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Convert to log scale (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Determine the current length of the Mel spectrogram
    current_length = S_dB.shape[1]

    if current_length < fixed_length:
        # Pad with zeros if the current length is less than the fixed length
        pad_width = fixed_length - current_length
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if the current length is more than the fixed length
        S_dB = S_dB[:, :fixed_length]

    return S_dB

class AudioDataset(Dataset):
    def __init__(self, dataframe, max_length=250):
        self.dataframe = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        file_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        
        # Load the audio file using librosa to apply mel spectrogram
        waveform, sr = librosa.load(file_path, sr=44100)
        
        # Compute the mel spectrogram
        mel_spectrogram = self._get_mel_spectrogram(waveform, sr)
        
        return torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0) , torch.tensor(label, dtype=torch.long)

    def _get_mel_spectrogram(self, waveform, sr):
        # Extract Mel spectrogram using the predefined function
        mel_spectrogram = get_fixed_length_mel_spectrogram(waveform, sr, n_mels=mel_dim, fixed_length=self.max_length)
        return mel_spectrogram
    
    
# Example function to create DataLoader (not part of the training loop yet)
def create_dataloader(dataframe, batch_size):
    dataset = AudioDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader