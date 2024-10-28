# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 05:22:39 2024

@author: Jelena
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

file_path = os.path.join('..','podaci', 'split-over results')

# make plots english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)
surprisal_list = ['Surprisal GPT-2', 'Surprisal Yugo', 'Surprisal ngram-3']
surprisal_colour = {'Surprisal GPT-2': (0, 0 , 1, 1), 
                    'Surprisal Yugo':(1, 0 , 0, 1),
                    'Surprisal ngram-3':(1, 0, 1, 1)}

x_axis = [0, 1, 2, 3, 4]

for surprisal in surprisal_list:
    df = pd.read_csv(os.path.join(file_path, f"{surprisal}_results.csv"))
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        for emotion in [0,1,2,3,4]:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis = []
            y_std  = []
            
            for k in x_axis:
                y_axis.append(emotion_data[emotion_data['k']==k]['y_axis'].values[0])
                y_std.append(emotion_data[emotion_data['k']==k]['y_std'].values[0])
                
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
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
fig.legend(['gpt-2','gpt-2 std', 'yugo', 'yugo std','3-gram','3-gram std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
        # Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()



fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)
surprisal_list = ['Surprisal BERT', 'Surprisal BERTic']
surprisal_colour = {'Surprisal BERT': (0, 0 , 1, 1), 
                    'Surprisal BERTic':(1, 0 , 0, 1)}


for surprisal in surprisal_list:
    df = pd.read_csv(os.path.join(file_path, f"{surprisal}_results.csv"))
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        for emotion in [0,1,2,3,4]:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis = []
            y_std  = []
            
            for k in x_axis:
                y_axis.append(emotion_data[emotion_data['k']==k]['y_axis'].values[0])
                y_std.append(emotion_data[emotion_data['k']==k]['y_std'].values[0])
                
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
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
fig.legend(['bert', 'bert std', 'bertic', 'bertic std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()



# make plots serbian
fig = plt.figure(figsize=(12,8))
emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
fig.suptitle('Утицај степена сурприсала на предикцију трајања изговора', fontsize=30)
surprisal_list = ['Surprisal GPT-2', 'Surprisal Yugo', 'Surprisal ngram-3']
surprisal_colour = {'Surprisal GPT-2': (0, 0 , 1, 1), 
                    'Surprisal Yugo':(1, 0 , 0, 1),
                    'Surprisal ngram-3':(1, 0, 1, 1)}


for surprisal in surprisal_list:
    df = pd.read_csv(os.path.join(file_path, f"{surprisal}_results.csv"))
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        for emotion in [0,1,2,3,4]:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis = []
            y_std  = []
            
            for k in x_axis:
                y_axis.append(emotion_data[emotion_data['k']==k]['y_axis'].values[0])
                y_std.append(emotion_data[emotion_data['k']==k]['y_std'].values[0])
                
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
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
fig.legend(['gpt-2','gpt-2 std', 'yugo', 'yugo std','3-gram','3-gram std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))              
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


# plot for bidirectional mdoels in serbian
fig = plt.figure(figsize=(12,8))
emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
fig.suptitle('Утицај степена сурприсала на предикцију трајања изговора', fontsize=30)
surprisal_list = ['Surprisal BERT', 'Surprisal BERTic']
surprisal_colour = {'Surprisal BERT': (0, 0 , 1, 1), 
                    'Surprisal BERTic':(1, 0 , 0, 1)}


for surprisal in surprisal_list:
    df = pd.read_csv(os.path.join(file_path, f"{surprisal}_results.csv"))
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        for emotion in [0,1,2,3,4]:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis = []
            y_std  = []
            
            for k in x_axis:
                y_axis.append(emotion_data[emotion_data['k']==k]['y_axis'].values[0])
                y_std.append(emotion_data[emotion_data['k']==k]['y_std'].values[0])
                
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
            c = surprisal_colour[surprisal]
            plt.scatter(x_axis, y_axis, s=100, color=c)
            # Add shadows based on the standard deviation of y_axis
            shadow_c = c[:-1] + (0.3,)
            y_std = np.std(y_axis)
            plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)

# Add a common x-axis label
fig.text(0.5, 0.001, 'степен сурприсала', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['bert', 'bert std', 'bertic', 'bertic std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()