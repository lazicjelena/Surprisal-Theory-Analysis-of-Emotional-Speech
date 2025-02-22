# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:31:47 2025

@author: Jelena
"""

from my_functions import inf_k_model, calculate_delta_ll_old
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import os
    
file_path = output_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)

columns = ['word', 'emotion', 'speaker', 'time', 'speaker gender', 'target sentence', 
           'log probability', 'length', 'surprisal GPT', 'fold', 'baseline']

df = df[columns]

# make plots english
emotions = ["neutral", "happy", "sad", "scared", "angry"]
#fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)
surprisal_list = ['surprisal GPT']


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
    
    for emotion in [0,1,2,3,4]:
        emotion_data = speaker_data[speaker_data['emotion'] == emotion]
        
        if speaker_data['speaker gender'].iloc[0] == 'm':
            surprisal_colour = {'surprisal GPT': (0, 0 , 1, 1)}
        else:
            surprisal_colour = {'surprisal GPT': (1, 0 , 0, 1)}
            
        plt.subplot(1,5, emotion + 1)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll_old(emotion_data, surprisal, k)
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
    fig.legend(['gpt', 'gpt std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.show()


# # make plots serbian
# fig = plt.figure(figsize=(12,8))
# emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
# #fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)

# x_axis = np.arange(0.25, 3, 0.25)
    
# for gender in ['f', 'm']:
#     gender_data = df[df['speaker gender'] == gender]
    
#     for emotion in [0,1,2,3,4]:
#         emotion_data = gender_data[gender_data['emotion'] == emotion]
#         if gender == 'f':
#             plt.subplot(2,5, emotion + 1)
#         else:
#             plt.subplot(2,5, emotion + 6)
            
#         # calculate model predictions
#         for surprisal in surprisal_list:
#             y_axis = []
#             y_std = []
            
#             for i in x_axis:
#                 k = round(i, 2)
#                 delta_element,  std_element = calculate_delta_ll_old(emotion_data, surprisal, k)
#                 y_axis.append(delta_element)
#                 y_std.append(std_element)
            
#             c = surprisal_colour[surprisal]
#             plt.scatter(x_axis, y_axis, s=100, color=c)
#             # Add shadows based on the standard deviation of y_axis
#             shadow_c = c[:-1] + (0.3,)
#             y_std = np.std(y_axis)
#             plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
#             plt.title(emotions[emotion], fontsize = 25)
#             plt.tick_params(axis='both', which='major', labelsize=15)

# # Add a common x-axis label
# fig.text(0.5, 0.001, 'степен сурприсала', ha='center', va='center', fontsize=25)
# # Add a common y-axis label
# fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
# fig.legend(['gpt', 'gpt std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))  # Adjust layout for better spacing
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.show()

