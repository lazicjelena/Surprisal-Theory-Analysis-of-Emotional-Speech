# -*- coding: utf-8 -*-
"""correlation_results_representation.py

Created on Sun May 12 17:25:29 2024
@author: Jelena
lazic.jelenaa@gmail.com

Ova skripta prikazuje rezultate korelacije parametara wavelet transform i 
surprisala razlicitih model. Rezultati se koriste u radu.

Pipeline role
-------------
Reporting script for the wavelet-prosody / surprisal correlation
analysis. Reads ``../podaci/prominence_data.csv`` (built by
``prominence_build_dataset.py``), derives a per-row
``relative duration = duration / len(word)`` column and prints
two diagnostic blocks: (1) per-(gender, emotion) mean prominence,
and (2) per-(emotion, surprisal model) Pearson correlation
between the surprisal score and the relative spoken-word duration
for the male speaker subset. Output is purely textual; no CSVs
or figures are written.
"""

import pandas as pd
import os
import numpy as np

data_path = os.path.join('..','podaci', 'prominence_data.csv') 
data = pd.read_csv(data_path)

relative_duration_list = []
for index, row in data.iterrows():
    duration = row['duration']
    length = len(row['word'])
    relative_duration_list.append(duration/length)

data['relative duration'] = relative_duration_list 


# Selecting the columns for correlation
columns_of_interest = ['surprisal BERT', 'surprisal BERTic', 'surprisal ngram3 alpha4', 'surprisal GPT', 'surprisal yugo']
emotions = [0, 1, 2, 3, 4]

for gender in ['f', 'm']:
    print(f"Gender: {gender}")
    df = data[data['gender']==gender]
    for emotion in [0,1,2,3,4]:
        emotion_df = df[df['emotion']==emotion]
        mean_value = np.mean(emotion_df['prominence'])
        print(f"Emotion {emotion}: {mean_value}")
        

gender_df = data[data['gender']=='m']
for column in columns_of_interest:
    print(f"For model {column} correlation values are following: ")
    for emotion in emotions:
        df = gender_df[gender_df['emotion']==emotion]
        correlation_value = df[column].corr(df['relative duration'])
        #print(f"{emotion} {correlation_value}")
        print(f"{correlation_value:.4f}")
        
