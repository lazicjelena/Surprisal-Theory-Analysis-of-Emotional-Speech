# -*- coding: utf-8 -*-
"""utils.audio_utils

Centralizovani audio-feature helperi koje koriste vise foldera projekta.

P-012 (Faza 2-C): cross-folder konsolidacija. Funkcija
``get_fixed_length_mel_spectrogram`` je prethodno postojala kao
byte-identicna kopija u 2 fajla (emotion_recognition/audio_utils,
mel_surprisal_analysis/calculate_mel_spectrum). Tijelo funkcije NIJE
mijenjano - samo premjesteno na jedno centralno mjesto (zero-change).

Napomena: ``extract_mel_spectrogram`` (iz emotion_recognition/) nije
ukljucen ovdje jer ima dependency na modulne globals (``mel_dim``,
``fixed_length``) koje se definisu u pozivajucem skripti. Njegova
konsolidacija zahtijeva posebnu odluku o ovim konstantama.

Pipeline role
-------------
Project-wide shared audio-feature helper module imported under the
package path ``utils.audio_utils``. Hosts
:func:`get_fixed_length_mel_spectrogram`, the pad-or-truncate
log-Mel spectrogram extractor consumed by ``audiodataset.py``
(training/validation feature pipeline) and
``prosody_parameters_and_mfcc.py`` (illustrative figures) in the
``emotion_recognition/`` chain, plus
``mel_surprisal_analysis/calculate_mel_spectrum.py`` which builds
mel-feature datasets for the surprisal-vs-mel analysis.
"""

import librosa
import numpy as np


def get_fixed_length_mel_spectrogram(y, sr, n_mels, fixed_length):
    """Compute a fixed-length log-Mel spectrogram from a waveform.

    The Mel spectrogram is converted to dB scale (``power_to_db``
    with ``ref=np.max``) and either zero-padded along the time
    axis (if it is shorter than ``fixed_length``) or
    right-truncated (if it is longer).

    Parameters
    ----------
    y : numpy.ndarray
        Mono waveform, typically as returned by :func:`librosa.load`.
    sr : int
        Sampling rate of ``y`` in Hz.
    n_mels : int
        Number of Mel bands.
    fixed_length : int
        Target number of time frames.

    Returns
    -------
    numpy.ndarray
        Log-Mel spectrogram of shape ``(n_mels, fixed_length)``.
    """
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
