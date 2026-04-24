# -*- coding: utf-8 -*-
"""audio_utils.py
Audio feature extraction pomocne funkcije izdvojene iz:
  - Emotion recognition/audiodataset.py
  - Emotion recognition/prosody_parameters_and_mfcc.py

P-008 (Faza 2-B): zajednicke IDENTICNO funkcije unutar foldera
'Emotion recognition/'. Tijelo funkcije NIJE mijenjano.

Napomena: get_fixed_length_mel_spectrogram postoji i u:
  - Mel coefficients and surprisals/calculate_mel_spectrum.py
Cross-folder konsolidacija nije dio P-008 - ostaje za P-009.

Takodje, extract_mel_spectrogram nije prebacen u ovaj utils jer ima
dependency na modulne globals (mel_dim, fixed_length) koje se definisu
u pozivajucem skripti. Njegova konsolidacija zahtijeva posebnu odluku
o ovim konstantama - ostaje za P-009.
"""

import librosa
import numpy as np


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
