# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:36:22 2025

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

# columns = ['word', 'emotion', 'time', 'speaker gender', 'target sentence', 
#            'log probability', 'length', 'surprisal BERT', 'surprisal BERTic', 
#            'surprisal RoBERTa', 'surprisal ALBERT', 'surprisal BERT uni',
#            'surprisal BERTic uni', 'surprisal RoBERTa uni', 'surprisal ALBERT uni',
#            'fold', 'baseline']

# df = df[columns]

# make plots english
fig = plt.figure(figsize=(12,8))

emotions = ["neutral", "happy", "sad", "scared", "angry"]
#fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)

if 1:
    surprisal_list = ['surprisal GPT', 
                      'surprisal yugo', 
                      'surprisal ngram3 alpha4'
                      ]
    surprisal_colour = {'surprisal GPT': (0, 0 , 1, 1), 
                        'surprisal yugo':(1, 0 , 0, 1),
                        'surprisal ngram3 alpha4':(1, 0, 1, 1)}
    legend = ['gpt', 'gpt std', 
              'yugo', 'yugo std', 
              'ngram3','ngram3 std']

k = 2
functions = ['power', 'linear','logarithmic']
bar_with = 0.25

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for function in functions:
        df = inf_k_model(df, k, surprisal, function = function)
warnings.resetwarnings()
    


for gender in ['f', 'm']:
    gender_data = df[df['speaker gender'] == gender]
    
    for emotion in [0,1,2,3,4]:
        x_axis = [0, 1, 2]
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            x_axis = [i + bar_with for i in x_axis]
            
            for function in functions:
                
                delta_element,_ = calculate_delta_ll_old(emotion_data, surprisal, k, function = function)
                y_axis.append(delta_element)
            
            c = surprisal_colour[surprisal]
            plt.bar(x_axis, y_axis, color=c, width=0.4)

            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=20)
            
        plt.xticks(x_axis, [r'$s^2$', r'$2s$', r'$\log(s)$'], fontsize=20, rotation=45)

# Add a common x-axis label
fig.text(0.5, 0.001, 'surprisal power', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(surprisal_list, fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


