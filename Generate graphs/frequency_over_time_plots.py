# -*- coding: utf-8 -*-
"""frequency_over_time_plots.py
Jelenina skripta
lazic.jelenaa@gmail.com

U ovoj skripti plotuje se grafik promjene frekvencije tokom vremena,
za svakog govornika i za svako emocionalno stanje.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def padding_sequence(f0_all_files):
    
    # Finding the maximum length of sublists
    max_length = max(len(sublist) for sublist in f0_all_files)
    # Pad each sublist individually
    padded_list = []
    for sublist in f0_all_files:
        padding = [np.nan] * (max_length - len(sublist))
        padded_sublist = np.concatenate((sublist, padding))
        padded_list.append(padded_sublist)
    
    return padded_list

file_path =  os.path.join('..','podaci', 'f0.csv')
df = pd.read_csv(file_path) 
df = df[df['user'] != '1052']
f0_array = [[]]

sr = 44100 # sampling rate

for i in range(0,len(df)):
    array_of_strings = [j for j in df['f0'][i].split()[1:][:-1]]
    array_of_floats = [float(val) if val.replace('.', '', 1).isdigit() else np.nan for val in array_of_strings]
    f0_array.append(array_of_floats)
    
padded_list = padding_sequence(f0_array[1:]) 
df['f0 values'] = padded_list

# speaker gender 
gender_file_path =  os.path.join('..','podaci', 'gender_data.csv')
gender_df = pd.read_csv(gender_file_path) 
df = pd.merge(df, gender_df, left_on='user', right_on='Speaker')


# make plots
emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
fig = plt.figure(figsize=(15,10))

for gender in ['m', 'f']:
    for emotion in [0,1,2,3,4]:
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
        gender_data = df[df['Gender'] == gender]
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        # Calculate the average of 'f0 values'
        data = [i for i in emotion_data['f0 values']]
        average_f0 = np.nanmean(data, axis = 0)
        std_f0 = np.nanstd(data, axis=0)
        
        # Create x-axis values (indices of the elements in the average_f0 list)
        x_values = np.arange(len(average_f0))
    
        # Plot the average_f0 values
        plt.plot(x_values, average_f0)
        plt.fill_between(x_values, average_f0 - std_f0, average_f0 + std_f0, color='blue', alpha=0.3, label='Standard deviation')
        
        plt.title(emotions[emotion], fontsize=20)
        plt.ylim([50,250])
        plt.xlim([50,500])
        
        
# Add a common x-axis label
fig.text(0.5, 0.07, 'одбирци', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.07, 0.5, 'f0 [Hz]', ha='center', va='center', rotation='vertical', fontsize=20)

