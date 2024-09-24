# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:57:31 2024

@author: Jelena
"""

import pandas as pd
import os

data_path = os.path.join('..','podaci', 'prominence_data.csv') 
data = pd.read_csv(data_path)
columns_of_interest = ['speaker', 'emotion', 'word', 'target sentence', 'duration', 'prominence', 'gender', 'surprisal GPT']
data = data[columns_of_interest]

def extraxt_parameter_over_emotion(data, parameter):
    
    print(f"Parameter: {parameter}")
    neutral_data = data[data['emotion'] == 0]
    
    for emotion in [1,2,3,4]:
        
        print(f"Emotional state: {emotion}")
        duration_list = []
        none_values = 0
        ind = 0
        last_sentence = 932947234
        words = []
        
        for _,row in neutral_data.iterrows():
            
            speaker = row['speaker']
            sentence = row['target sentence']
            if sentence != last_sentence:
                last_sentence = sentence 
                words = []
            word = row['word']
            words.append(word)
            
            search = data[data['emotion']==emotion]
            search = search[search['target sentence']==sentence]
            search = search[search['speaker']==speaker]
            search = search[search['word']==word]
            
            if len(search) > 1:
                index = words.count(word)-1
                duration = search[parameter].values[index]
                ind += 1
            else:
                if len(search)==1:
                    duration = search[parameter].values[0]
                else:
                    duration = 487923472842101
                    none_values += 1
            duration_list.append(duration)
            
        neutral_data[emotion] = duration_list
        neutral_data = neutral_data[neutral_data[emotion] != 487923472842101]
        print(f"None values count: {none_values}")
        print(f"Double words count: {ind}")
        
    print(f"Final number of words: {len(neutral_data)}")
    
    return neutral_data

data_duration = extraxt_parameter_over_emotion(data, 'prominence')

# Plot 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

emotions = ['neutral', 'happy', 'sad', 'scared', 'angry']
# Assuming data_duration is your DataFrame and emotion is a list of emotions [1,2,3,4]
fig, axs = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Frequency Modulations over Different Emotional States', fontsize=30)

for idx, emotion in enumerate([1, 2, 3, 4]):
    # Plot for female speakers
    ax_f = axs[0, idx]
    df_f = data_duration[data_duration['gender'] == 'f']
    ax_f.scatter(df_f['prominence'], df_f[emotion], color='r', label='Female')

    # Fit line and calculate MSE for female speakers
    coeffs_f = np.polyfit(df_f['prominence'], df_f[emotion], 1)
    k_f, n_f = coeffs_f
    line_f = np.polyval(coeffs_f, df_f['prominence'])
    mse_f = mean_squared_error(df_f[emotion], line_f)
    ax_f.plot(df_f['prominence'], line_f, color='r', linestyle='--', 
              label=f'Fit: y={k_f:.2f}x+{n_f:.2f}\nMSE: {mse_f:.2f}')

    ax_f.set_title(f'{emotions[emotion]}', fontsize=25)
    ax_f.legend(fontsize=20)
    
    # Plot for male speakers
    ax_m = axs[1, idx]
    df_m = data_duration[data_duration['gender'] == 'm']
    ax_m.scatter(df_m['prominence'], df_m[emotion], color='b', label='Male')

    # Fit line and calculate MSE for male speakers
    coeffs_m = np.polyfit(df_m['prominence'], df_m[emotion], 1)
    k_m, n_m = coeffs_m
    line_m = np.polyval(coeffs_m, df_m['prominence'])
    mse_m = mean_squared_error(df_m[emotion], line_m)
    ax_m.plot(df_m['prominence'], line_m, color='b', linestyle='--', 
              label=f'Fit: y={k_m:.2f}x+{n_m:.2f}\nMSE: {mse_m:.2f}')

    ax_m.set_title(f'{emotions[emotion]}', fontsize=25)
    ax_m.legend(fontsize=20)

# Set x-axis label only in the middle of each row
for ax in axs[0, :]:
    ax.tick_params(axis='x', which='major', labelsize=20)

for ax in axs[1, :]:
    ax.tick_params(axis='x', which='major', labelsize=20)
    
# Increase the y-axis numbers' size for all subplots
for ax in axs.flat:
    ax.tick_params(axis='y', which='major', labelsize=20)  # Adjust labelsize as needed

# Add x-axis labels only in the middle
fig.text(0.5, 0.02, 'Neutral Speech', ha='center', va='center', fontsize=25)
fig.text(0.02, 0.5, 'Emotional Speech', ha='center', va='center', rotation='vertical', fontsize=25)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
