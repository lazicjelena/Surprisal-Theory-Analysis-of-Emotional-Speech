# -*- coding: utf-8 -*-
"""frequency_over_time.py
Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta sluzi izdvajanju promjene frekvencije usrednjene po govorniku i emocijama.

Pipeline role
-------------
Acoustic feature extractor for the F0-trajectory plots. Walks every
WAV file under ``../podaci/data_mono/<speaker>/<emotion>/``, runs
:func:`librosa.pyin` with gender-conditioned ``fmin``/``fmax`` (read
from ``../podaci/gender_data.csv``) to estimate frame-level F0, pads
all per-file F0 sequences to a common length via
:func:`generate_graphs_utils.padding_sequence`, averages frame-wise
across files for that ``(speaker, emotion)`` cell, and accumulates
one row per ``(speaker, emotion)`` into ``../podaci/f0.csv``. That
CSV is the input to ``frequency_over_time_plots.py``.
"""

import numpy as np
import librosa
import pandas as pd
import os

from generate_graphs.generate_graphs_utils import padding_sequence


user_list = []
emotion_list = []
f0_av_user_emotion = [[]]

# Define the directory path
folder_directory_path = os.path.join('..','podaci', 'data_mono')
transcript_directory_path = os.path.join('..','podaci', 'transcript_corrected')
gender_directory_path = os.path.join('..','podaci', 'gender_data.csv')
gender_df = pd.read_csv(gender_directory_path)

# Loop through the directory
for user in os.listdir(folder_directory_path):
    print(user)
    for emotion in [0,1,2,3,4]:
        directory_path = os.path.join(folder_directory_path, user, str(emotion))    
        if os.path.exists(directory_path):
            print(emotion)
            # Lists to store F0 values and corresponding times for all files
            f0_all_files = [[]]
            for filename in os.listdir(directory_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(directory_path, filename)
                    # Load the audio file
                    y, sr = librosa.load(file_path, sr=None)
                    # Calculate F0 (pitch)
                    if gender_df[gender_df['Speaker'] == int(user)]['Gender'].values == 'm':
                        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=90, fmax=155, sr=44100, frame_length=1024)
                    else:
                        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=165, fmax=255, sr=44100, frame_length=1024)
                    # Append F0 values and corresponding times
                    f0_all_files.append(f0)
                    
            # Calculate average F0 over time, excluding NaN values
            padded_list = padding_sequence(f0_all_files)
            average_f0 = np.nanmean(padded_list, axis=0)
            
            # keep data for df
            user_list.append(user)
            emotion_list.append(emotion)
            f0_av_user_emotion.append(average_f0)
    
            #time = np.arange(len(average_f0)) * librosa.samples_to_time(1, sr=sr)
    
    # Create DataFrame
    df = pd.DataFrame({'user': user_list, 'emotion': emotion_list, 'f0': f0_av_user_emotion[1:]})
    output_path = os.path.join('..','podaci', 'f0.csv')
    df.to_csv(output_path, index=False) 
