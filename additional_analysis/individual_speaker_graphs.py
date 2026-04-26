# -*- coding: utf-8 -*-
"""individual_speaker_graphs.py

Created on Sat Feb 22 12:31:47 2025

@author: Jelena

Pipeline role
-------------
Per-speaker visualisation driver for the post-recension
duration regression. Loads ``../podaci/training_data.csv``,
sweeps the surprisal exponent ``k`` via
:func:`my_functions.inf_k_model` for every surprisal in the
configured list (GPT, Yugo, ngram-3 alpha-4) and plots, per
emotion and per speaker, the delta log-likelihood improvement
over the surprisal-free baseline (via
:func:`my_functions.calculate_delta_ll`). Output is purely
visual; no CSVs are written.
"""

from additional_analysis.my_functions import inf_k_model, calculate_delta_ll
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import os
    
file_path = output_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)

# make plots english
emotions = ["neutral", "happy", "sad", "scared", "angry"]
#fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)

if 1:
    surprisal_list = ['surprisal GPT', 
                      'surprisal yugo', 
                      'surprisal ngram3 alpha4']
    surprisal_colour = {'surprisal GPT': (0, 0 , 1, 1), 
                        'surprisal yugo':(1, 0 , 0, 1),
                        'surprisal ngram3 alpha4':(1, 0, 1, 1)}
    legend_list = ['gpt', 'gpt std',
                   'yugo', 'yugo std',
                   'ngram3', 'ngram3 std']


x_axis = np.arange(0.25, 3, 0.25)

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for i in x_axis:
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal)
warnings.resetwarnings()
    
for speaker in df['speaker'].unique():
    
    fig = plt.figure(figsize=(12,4))
    
    speaker_data = df[df['speaker'] == speaker]
    title = f"Speaker number: {speaker_data['speaker'].iloc[0]}, Speaker Gender: "
    if speaker_data['speaker gender'].iloc[0] == 'm':
        title += 'Male'
    else:
        title += 'Female'
    
    if speaker_data['speaker gender'].iloc[0] == 'm':
        continue 
    
    fig.suptitle(title, fontsize=30)
            
    for emotion in [0,1,2,3,4]:
        emotion_data = speaker_data[speaker_data['emotion'] == emotion]
            
        plt.subplot(1,5, emotion + 1)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll(mode="prominence", data=emotion_data, surprisal_name=surprisal, k=k)
                y_axis.append(delta_element)
                y_std.append(std_element)
            
            c = surprisal_colour[surprisal]
            plt.scatter(x_axis, y_axis, s=100, color=c)
            # Add shadows based on the standard deviation of y_axis
            shadow_c = c[:-1] + (0.3,)
            y_std = np.std(y_axis)
            plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)

    # Add a common x-axis label
    fig.text(0.5, 0.001, 'surprisal power', ha='center', va='center', fontsize=25)
    # Add a common y-axis label
    fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
    fig.legend(legend_list, fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.show()



