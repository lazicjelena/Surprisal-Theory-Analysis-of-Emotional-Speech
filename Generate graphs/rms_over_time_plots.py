# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 08:25:31 2024

@author: Jelena
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

file_path =  os.path.join('..','podaci', 'rms.csv')
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
fig = plt.figure(figsize=(12,8))
fig.suptitle('Промјена RMS Енергије говора', fontsize=30)


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
        if gender == 'f':
            plt.plot(x_values, average_f0,color = 'red')
            plt.fill_between(x_values, average_f0 - std_f0, average_f0 + std_f0, color='red', alpha=0.3, label='Standard deviation')
        else:
            plt.plot(x_values, average_f0,color = 'blue')
            plt.fill_between(x_values, average_f0 - std_f0, average_f0 + std_f0, color='blue', alpha=0.3, label='Standard deviation')
        
        plt.title(emotions[emotion], fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=15)
        if gender == 'f':
            plt.ylim([0,0.2])
        else:
            plt.ylim([0,0.2])
        plt.xlim([50,400])
        
        
# Add a common x-axis label
fig.text(0.5, 0.001, 'временски прозори', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, 'RMS Енергија', ha='center', va='center', rotation='vertical', fontsize=25)
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.grid(False)
plt.show()

# make plots english
emotions = ["neutral", "happy", "sad", "scared", "angry"]

fig, ax = plt.subplots(2,5, figsize=(12,8))
fig.suptitle('RMS Energy over Different Emotiononal States', fontsize=30)


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
        if gender == 'f':
            plt.plot(x_values, average_f0,color = 'red')
            plt.fill_between(x_values, average_f0 - std_f0, average_f0 + std_f0, color='red', alpha=0.3, label='Standard deviation')
        else:
            plt.plot(x_values, average_f0,color = 'blue')
            plt.fill_between(x_values, average_f0 - std_f0, average_f0 + std_f0, color='blue', alpha=0.3, label='Standard deviation')
    
    
        plt.title(emotions[emotion], fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=15)
        if gender == 'f':
            plt.ylim([0,0.2])
        else:
            plt.ylim([0,0.2])
        plt.xlim([50,400])


# Add a common x-axis label
fig.text(0.5, 0.001, 'time frames', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, 'RMS Energy', ha='center', va='center', rotation='vertical', fontsize=25)
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.grid(False)
plt.show()


for gender in ['m', 'f']:
    print(f"Gender: {gender}")
    for emotion in [0,1,2,3,4]:
        gender_data = df[df['Gender'] == gender]
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        # Calculate the average of 'f0 values'
        data = [i for i in emotion_data['f0 values']]
        average_f0 = np.nanmean(data, axis = 0)
        std_f0 = np.nanstd(data, axis=0)
        print(f"{np.nanmean(average_f0, axis=0):.3f} ({np.nanstd(average_f0, axis=0):.3f})")
