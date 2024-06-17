# -*- coding: utf-8 -*-
"""calculate_mel_spectrum.py

@author: Jelena

Ova skripta racuna mel spektar svih podataka da bi kasnije lakse moglo da se vrsi
obucavanje neuralne mreze.

"""

# pip install tqdm

import pandas as pd
from ast import literal_eval
import os
import numpy as np
import librosa
from tqdm import tqdm  # Import tqdm for progress bar

df_path = os.path.join('..','podaci','general_data.csv')
df = pd.read_csv(df_path)

# Convert string representations of lists to actual lists using literal_eval
df['Text random'] = df['Text random'].apply(literal_eval)
df['Text keras'] = df['Text keras'].apply(literal_eval)
df['Surprisal Values'] = df['Surprisal Values'].apply(literal_eval)

df.info()

vocab_size = 276
mel_dim = 80
fixed_length = 250

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


# Function to extract mel spectrogram from audio file
def extract_mel_spectrogram(audio_file):
    # Load audio file and extract mel spectrogram using Librosa
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = get_fixed_length_mel_spectrogram(y, sr, n_mels=mel_dim, fixed_length=fixed_length)
    return mel_spectrogram


# Function to preprocess Mel spectrograms
def preprocess_mel_spectrum(audio_file):
    # Your preprocessing code here
    mel_spectrum = extract_mel_spectrogram(audio_file)
    return mel_spectrum

# Add a new column for Mel Spectrogram with progress bar
tqdm.pandas(desc="Preprocessing Mel Spectrogram")
df['Mel Spectrum'] = df['Audio Files'].progress_apply(preprocess_mel_spectrum)

pkl_path = os.path.join('..','podaci','general_data.pkl')
df.to_pickle(pkl_path)