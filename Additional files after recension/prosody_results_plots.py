# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:43:40 2025

@author: Jelena
"""


from my_functions import inf_k_model, calculate_delta_ll_old
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import os
    

prosody = 'energy'
file_path =  os.path.join('..','podaci', f"{prosody}_data.csv")
df = pd.read_csv(file_path)

columns = ['word', 'emotion', prosody, 'gender', 'target sentence', 
           'log probability', 'length', 'surprisal BERT', 'surprisal BERTic', 
           'surprisal GPT', 'surprisal yugo', 'surprisal ngram3 alpha4',
           'fold', 'baseline']

df = df[columns]

# make plots english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
#fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)

if 1:
    surprisal_list = ['surprisal BERT', 
                      'surprisal BERTic']
    surprisal_colour = {'surprisal BERT': (0, 0 , 1, 1), 
                        'surprisal BERTic':(1, 0 , 0, 1)}
    legend_list = ['bert', 'bert std',
                   'bertic', 'bertic std']
else:
    surprisal_list = ['surprisal ngram3 alpha4', 
                      'surprisal GPT', 
                      'surprisal yugo']
    surprisal_colour = {'surprisal ngram3 alpha4': (0, 0 , 1, 1), 
                        'surprisal GPT':(1, 0 , 0, 1),
                        'surprisal yugo':(1, 0, 1, 1)}
    legend_list = ['ngram3', 'ngram3 std',
                   'gpt', 'gpt std',
                   'yugo', 'yugo std']
    
x_axis = np.arange(0.25, 3, 0.25)

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for i in x_axis:
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal, prosody)
warnings.resetwarnings()
    
for gender in ['f', 'm']:
    gender_data = df[df['gender'] == gender]
    
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll_old(emotion_data, surprisal, k, prosody)
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

