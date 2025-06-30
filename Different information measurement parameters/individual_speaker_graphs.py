# -*- coding: utf-8 -*-
"""
Created on Fri May 23 23:33:00 2025

@author: Jelena
"""

from my_functions import add_column_with_surprisal, calculate_delta_ll
import matplotlib.pyplot as plt
#import warnings
import numpy as np
import pandas as pd
import os

''' Read data '''

gender = 'm'

# model = 'roberta'
# surprisal = 'surprisal RoBERTa uni'

# model = 'albert'
# surprisal = 'surprisal ALBERT uni'

model = 'bertic'
surprisal = 'surprisal BERTic uni'

model = 'bert'
surprisal = 'surprisal BERT uni'

model = 'gpt'
surprisal = 'surprisal GPT'

baseline_model = 'baseline -3'
#baseline_model = '{} model'.format(surprisal)

file_path =  os.path.join('..','podaci','transformer layers parameters', 'datasets', f"{model}.csv")
df = pd.read_csv(file_path)


''' make plots english '''

emotion_colour = {
    0: (1, 0, 0, 1),       # red
    1: (0, 0, 1, 1),       # blue
    2: (0, 1, 0, 1),       # green
    3: (1, 0.65, 0, 1),    # orange (RGB ≈ 255,165,0)
    4: (0.5, 0, 0.5, 1)    # purple (RGB ≈ 128,0,128)
}


legend_list = ['neutral', 'neutral std',
               'happy', 'happy std',
               'sad', 'sad std',
               'scared', 'scared std',
               'angry', 'angry std']


parameters = [f'CE {j}' for j in range(1, 13)]
for parameter in parameters:
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
    

x_axis = np.arange(1, 13, 1)
    
for speaker in df['speaker'].unique():
    
    fig = plt.figure(figsize=(12,8))
    
    speaker_data = df[df['speaker'] == speaker]
    title = f"Speaker number: {speaker_data['speaker'].iloc[0]}, Speaker Gender: "
    if speaker_data['speaker gender'].iloc[0] == 'm':
        title += 'Male'
    else:
        title += 'Female'
    
    if speaker_data['speaker gender'].iloc[0] != gender:
        continue 
    
    fig.suptitle(title, fontsize=30)
        
    for emotion in [0,1,2,3,4]:
        emotion_data = speaker_data[speaker_data['emotion'] == emotion]
        
        y_axis = []
        y_std = []
        
        for parameter in parameters:
            
            delta_element, std_element = calculate_delta_ll(emotion_data,  f"{surprisal} {parameter} model", baseline_model)
            y_axis.append(delta_element)
            y_std.append(std_element)
            
        c = emotion_colour[emotion]
        #plt.scatter(x_axis, y_axis, s=100, color=c)
        plt.plot(x_axis, y_axis, linewidth =3, color=c)
        # Add shadows based on the standard deviation of y_axis
        shadow_c = c[:-1] + (0.3,)
        y_std = np.std(y_axis)
        plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
        plt.tick_params(axis='both', which='major', labelsize=15)

    # Add a common x-axis label
    fig.text(0.5, 0.001, 'layer', ha='center', va='center', fontsize=25)
    # Add a common y-axis label
    fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
    fig.legend(legend_list, fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.show()
