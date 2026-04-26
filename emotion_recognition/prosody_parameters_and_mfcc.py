# -*- coding: utf-8 -*-
"""prosody_parameters_and_mfcc.py

Created on Tue Sep 24 06:05:02 2024

@author: Jelena

In this file relationship between changes in prosodic parameters and mel coefficients
is representend. This is important because many emotion recognition papers on SEAC
dataset use mfcc as an input parameters.

Pipeline role
-------------
Illustrative script that motivates Mel-spectrogram input by
showing how a controlled prosodic perturbation (per-segment
speed-up / slow-down) changes a single SEAC utterance's log-Mel
spectrogram. Loads
``../podaci/data_mono/0001/0/1_0_0_MMMD01L__25-02-21-10-30-27.wav``,
splits it into 5 equal pieces, randomly stretches each piece
between 50 percent and 150 percent of its original length via
:func:`audio_duration_modification`, concatenates the result and
writes it to ``modified_audio.wav``. The original and modified
log-Mel spectrograms are then plotted side by side in both
English and Cyrillic Serbian variants.
"""

from pydub import AudioSegment
from pydub.effects import speedup
import librosa.display
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os

from utils.audio_utils import get_fixed_length_mel_spectrogram

mel_dim = 80
fixed_length = 250
sr = 44100


# Function to extract mel spectrogram from audio file
def extract_mel_spectrogram(audio_file):
    """Load a WAV file and return its fixed-length log-Mel spectrogram.

    Wraps :func:`librosa.load` (with the file's native sampling
    rate) and the shared
    :func:`audio_utils.get_fixed_length_mel_spectrogram` using the
    module-level ``mel_dim`` and ``fixed_length`` constants.

    Parameters
    ----------
    audio_file : str
        Path to a WAV file.

    Returns
    -------
    numpy.ndarray
        Log-Mel spectrogram of shape
        ``(mel_dim, fixed_length)``.
    """
    # Load audio file and extract mel spectrogram using Librosa
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = get_fixed_length_mel_spectrogram(y, sr, n_mels=mel_dim, fixed_length=fixed_length)
    return mel_spectrogram


# Function to randomly change the duration of an audio piece
def audio_duration_modification(audio_segment):
    """Randomly stretch one ``pydub`` audio segment.

    The target duration is sampled uniformly between 50 percent
    and 150 percent of the original length (with a minimum of 1
    sample). When the target is shorter, the segment is sped up
    via :meth:`pydub.AudioSegment.speedup`; when longer, the
    segment is slowed down by spawning a copy at a reduced frame
    rate and resetting the frame rate back to the original.

    Parameters
    ----------
    audio_segment : pydub.AudioSegment
        Input audio segment.

    Returns
    -------
    pydub.AudioSegment
        Time-stretched copy of ``audio_segment``.
    """
    original_duration = len(audio_segment)
    print(f"Original duration: {original_duration/44100}")    
    
    # Ensure target duration is between 50% and 150% of the original duration, avoiding zero duration
    min_target_duration = max(1, original_duration * 0.5)  # Minimum target duration is 1 ms
    max_target_duration = original_duration * 1.5
    target_duration = np.random.uniform(min_target_duration, max_target_duration)
    print(f"Target duration: {target_duration/44100}")
    # Calculate the ratio of target to original duration
    speed_ratio = original_duration / target_duration

    if speed_ratio > 1:
        # Speed up the audio if the target duration is shorter
        modified_audio = audio_segment.speedup(playback_speed=speed_ratio)
    else:
        # Slow down the audio if the target duration is longer
        modified_audio = audio_segment._spawn(audio_segment.raw_data, overrides={
            "frame_rate": int(audio_segment.frame_rate * speed_ratio)
        }).set_frame_rate(audio_segment.frame_rate)

    return modified_audio

def modify_audio(audio_file):
    """Apply a per-segment random speed perturbation to a WAV file.

    The input WAV is loaded as a :class:`pydub.AudioSegment`,
    split into 5 equal pieces, each piece is independently
    stretched via :func:`audio_duration_modification`, the pieces
    are concatenated and the result is written to
    ``modified_audio.wav``.

    Parameters
    ----------
    audio_file : str
        Path to the input WAV file.

    Returns
    -------
    str
        Path of the written ``modified_audio.wav``.
    """
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file)
    # Split the audio into 10 equal parts
    num_parts = 5
    duration_per_part = len(audio) // num_parts
    audio_parts = [audio[i * duration_per_part: (i + 1) * duration_per_part] for i in range(num_parts)]
    
    # Change duration for each audio part
    modified_parts = [audio_duration_modification(part) for part in audio_parts]
    
    # Concatenate the modified audio parts
    concatenated_audio = sum(modified_parts)
    
    # Specify a file name for the saved WAV file
    output_wav_file = 'modified_audio.wav'
    
    # Save the concatenated audio as a WAV file
    concatenated_audio.export(output_wav_file, format='wav')
    
    return output_wav_file


# Read original audio file
audio_file = os.path.join('..','podaci','data_mono', '0001', '0', '1_0_0_MMMD01L__25-02-21-10-30-27.wav')
# Calulate mel spectrum
S_original = extract_mel_spectrogram(audio_file)


# Modify audio file
output_wav_file = modify_audio(audio_file)
# Calculate mel spectrum
S_modified  = extract_mel_spectrogram(output_wav_file)


# Plot the mel spectrogram
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(S_original, sr=sr, x_axis='frames', y_axis = 'mel', fmax=8000)
plt.ylabel('amplitude [Hz]', fontsize = 15)
plt.xlabel('frame [/]', fontsize = 15)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Original File', fontsize = 20)
plt.subplot(2, 1, 2)
librosa.display.specshow(S_modified, sr=sr, x_axis='frames', y_axis = 'mel', fmax=8000)
plt.ylabel('amplitude [Hz]', fontsize = 15)
plt.xlabel('frame [/]', fontsize = 15)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Modified File', fontsize = 20)
plt.tight_layout()
plt.show()


# Plot the mel spectrogram
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(S_original, sr=sr, x_axis='frames', y_axis = 'mel', fmax=8000)
plt.ylabel('амплитуда [Hz]', fontsize = 15)
plt.xlabel('прозор [/]', fontsize = 15)
plt.colorbar(format='%+2.0f dB')
plt.title('Мел Спектрограм Оригиналног Сигнала', fontsize = 20)
plt.subplot(2, 1, 2)
librosa.display.specshow(S_modified, sr=sr, x_axis='frames', y_axis = 'mel', fmax=8000)
plt.ylabel('амплитуда [Hz]', fontsize = 15)
plt.xlabel('прозор [/]', fontsize = 15)
plt.colorbar(format='%+2.0f dB')
plt.title('Мел Спектрограм Модификованог Снимка', fontsize = 20)
plt.tight_layout()
plt.show()

