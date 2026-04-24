# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:18:26 2024

@author: Jelena
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torch

from audio_utils import get_fixed_length_mel_spectrogram


mel_dim = 80

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