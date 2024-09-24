# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 06:05:02 2024

@author: Jelena

In this file relationship between changes in prosodic parameters and mel coefficients
is representend. This is important because many emotion recognition papers on SEAC
dataset use mfcc as an input parameters.
"""

from pydub import AudioSegment
from pydub.effects import speedup
import librosa.display
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os 

mel_dim = 80
fixed_length = 250
sr = 44100

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


# Function to randomly change the duration of an audio piece
def audio_duration_modification(audio_segment):
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

