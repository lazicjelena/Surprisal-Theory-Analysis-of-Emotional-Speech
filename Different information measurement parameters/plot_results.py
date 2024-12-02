# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:29:12 2024

@author: Jelena
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

file_path = os.path.join('..','podaci','information measurements parameters', "results.csv") 

# make plots english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Surprisal vs Contextual Entropy', fontsize=30)
parameters = ['Surprisal GPT-2',
             'Contextual Entropy',
             'Surprisal GPT-2 Contextual Entropy'
             ]

parameter_colour = {'Surprisal GPT-2': (1, 0 , 1, 1), 
                    'Contextual Entropy':(1, 0 , 0, 1),
                    'Surprisal GPT-2 Contextual Entropy':(0, 0, 1, 1)}

gender_list = {'f':'female', 'm':'male'}

x_axis = [0, 1, 2, 3, 4]
emotion_list = emotions

for p in parameters:
    df = pd.read_csv(file_path)
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        y_axis = []
        y_std  = []
            
        for emotion in x_axis:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis.append(emotion_data[emotion_data['parameter']==p]['y_axis'].values[0])
            y_std.append(emotion_data[emotion_data['parameter']==p]['y_std'].values[0])
                
        if gender == 'f':
            plt.subplot(1,2,1)
        else:
            plt.subplot(1,2,2)
            
        c = parameter_colour[p]
        plt.scatter(x_axis, y_axis, s=100, color=c)
        # Add shadows based on the standard deviation of y_axis
        shadow_c = c[:-1] + (0.3,)
        y_std = np.std(y_axis)
        plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
  
            
        #plt.xticks([])
        plt.title(gender_list[gender], fontsize = 25)
        plt.xticks(ticks=x_axis, labels=emotion_list, rotation=45, fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=15)
        
        
# Add a common x-axis label
fig.text(0.5, 0.001, '', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['Surprisal GPT-2', 'Surprprisal GPT-2 std',
            'Contextual Entropy', 'Contextual Entropy std',
            'Surprisal GPT-2 + Contextual Entropy', 'Surprisal GPT-2 + Contextual Entropy std'],
           fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


''' Information values '''

# make plots english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Information Value Parameters', fontsize=30)
parameters = ['Surprisal GPT-2 Context Embedding',
              'Surprisal GPT-2 Non-context Embedding',
              'Surprisal GPT-2 Pos-Tag',
              'Surprisal GPT-2 Orthographic'
             ]

parameter_colour = {'Surprisal GPT-2 Context Embedding': (1, 0 , 1, 1), 
                    'Surprisal GPT-2 Non-context Embedding':(1, 0 , 0, 1),
                    'Surprisal GPT-2 Pos-Tag':(0, 0, 1, 1),
                    'Surprisal GPT-2 Orthographic':(0, 1, 0, 1)}

gender_list = {'f':'female', 'm':'male'}

x_axis = [0, 1, 2, 3, 4]
emotion_list = ['neutral',
                'happy',
                'sad',
                'scared',
                'angry']

for p in parameters:
    df = pd.read_csv(file_path)
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        y_axis = []
        y_std  = []
            
        for emotion in x_axis:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis.append(emotion_data[emotion_data['parameter']==p]['y_axis'].values[0])
            y_std.append(emotion_data[emotion_data['parameter']==p]['y_std'].values[0])
                
        if gender == 'f':
            plt.subplot(1,2,1)
        else:
            plt.subplot(1,2,2)
            
        c = parameter_colour[p]
        plt.scatter(x_axis, y_axis, s=100, color=c)
        # Add shadows based on the standard deviation of y_axis
        shadow_c = c[:-1] + (0.3,)
        y_std = np.std(y_axis)
        plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
  
            
        #plt.xticks([])
        plt.title(gender_list[gender], fontsize = 25)
        plt.xticks(ticks=x_axis, labels=emotion_list, rotation=45, fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=15)
        
        
# Add a common x-axis label
fig.text(0.5, 0.001, '', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['Context Embedding', 'Context Embedding std',
            'Non-context Embedding', 'Non-context Embedding std',
            'Pos-Taging', 'Pos-Taging std',
            'Orthographic', 'Orthographic std'],
           fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


''' Adjusted Surprisals '''

# make plots english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Adjusted Surprisal Parameters', fontsize=30)
parameters = ['Surprisal GPT-2 AS Context Embedding',
              'Surprisal GPT-2 AS Non-context Embedding',
              'Surprisal GPT-2 AS Pos-Tag',
              'Surprisal GPT-2 AS Orthographic'
             ]

parameter_colour = {'Surprisal GPT-2 AS Context Embedding': (1, 0 , 1, 1), 
                    'Surprisal GPT-2 AS Non-context Embedding':(1, 0 , 0, 1),
                    'Surprisal GPT-2 AS Pos-Tag':(0, 0, 1, 1),
                    'Surprisal GPT-2 AS Orthographic':(0, 1, 0, 1)}

gender_list = {'f':'female', 'm':'male'}

x_axis = [0, 1, 2, 3, 4]
emotion_list = ['neutral',
                'happy',
                'sad',
                'scared',
                'angry']

for p in parameters:
    df = pd.read_csv(file_path)
                                  
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
    
        y_axis = []
        y_std  = []
            
        for emotion in x_axis:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            y_axis.append(emotion_data[emotion_data['parameter']==p]['y_axis'].values[0])
            y_std.append(emotion_data[emotion_data['parameter']==p]['y_std'].values[0])
                
        if gender == 'f':
            plt.subplot(1,2,1)
        else:
            plt.subplot(1,2,2)
            
        c = parameter_colour[p]
        plt.scatter(x_axis, y_axis, s=100, color=c)
        # Add shadows based on the standard deviation of y_axis
        shadow_c = c[:-1] + (0.3,)
        y_std = np.std(y_axis)
        plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
  
            
        #plt.xticks([])
        plt.title(gender_list[gender], fontsize = 25)
        plt.xticks(ticks=x_axis, labels=emotion_list, rotation=45, fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=15)
        
        
# Add a common x-axis label
fig.text(0.5, 0.001, '', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['Context Embedding', 'Context Embedding std',
            'Non-context Embedding', 'Non-context Embedding std',
            'Pos-Taging', 'Pos-Taging std',
            'Orthographic', 'Orthographic std'],
           fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


