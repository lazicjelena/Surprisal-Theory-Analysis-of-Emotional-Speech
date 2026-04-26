# -*- coding: utf-8 -*-
"""calculate_mel_spectrum.py

@author: Jelena

Ova skripta racuna mel spektar svih podataka da bi kasnije lakse moglo da se vrsi
obucavanje neuralne mreze.

Pipeline role
-------------
Mel-spectrogram precomputation step in the surprisal/mel pipeline.
Loads ``../podaci/general_data.csv`` (built by
``build_dataset.py``), parses the literal-encoded
``Text random``, ``Text keras`` and ``Surprisal Values`` list
columns back into Python lists, then for every WAV path in
``Audio Files`` computes a fixed-length log-Mel spectrogram
(80 mel bands x 250 frames; pad-or-truncate). The augmented frame
is serialized to ``../podaci/general_data.pkl``; ``model.py``
consumes this pickle directly so spectrogram I/O is not on the
training loop's hot path.
"""

# pip install tqdm

import pandas as pd
from ast import literal_eval
import os
import numpy as np
import librosa
from tqdm import tqdm  # Import tqdm for progress bar
from utils.audio_utils import get_fixed_length_mel_spectrogram

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

# Function to extract mel spectrogram from audio file
def extract_mel_spectrogram(audio_file):
    """Load a WAV file and return its fixed-length log-Mel spectrogram.

    Wraps :func:`librosa.load` (with the file's native sampling
    rate) and :func:`get_fixed_length_mel_spectrogram` using the
    module-level ``mel_dim`` and ``fixed_length`` constants.

    Parameters
    ----------
    audio_file : str
        Path to a WAV file.

    Returns
    -------
    numpy.ndarray
        Log-Mel spectrogram of shape ``(mel_dim, fixed_length)``.
    """
    # Load audio file and extract mel spectrogram using Librosa
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = get_fixed_length_mel_spectrogram(y, sr, n_mels=mel_dim, fixed_length=fixed_length)
    return mel_spectrogram


# Function to preprocess Mel spectrograms
def preprocess_mel_spectrum(audio_file):
    """Thin wrapper around :func:`extract_mel_spectrogram`.

    Kept as a separate function because it is the entry point used
    with :meth:`pandas.Series.progress_apply` for tqdm progress
    reporting on the full dataset.

    Parameters
    ----------
    audio_file : str
        Path to a WAV file.

    Returns
    -------
    numpy.ndarray
        Log-Mel spectrogram of shape ``(mel_dim, fixed_length)``.
    """
    # Your preprocessing code here
    mel_spectrum = extract_mel_spectrogram(audio_file)
    return mel_spectrum

# Add a new column for Mel Spectrogram with progress bar
tqdm.pandas(desc="Preprocessing Mel Spectrogram")
df['Mel Spectrum'] = df['Audio Files'].progress_apply(preprocess_mel_spectrum)

pkl_path = os.path.join('..','podaci','general_data.pkl')
df.to_pickle(pkl_path)