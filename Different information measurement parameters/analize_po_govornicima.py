# -*- coding: utf-8 -*-
"""
Created on Fri May 23 23:46:00 2025

@author: Jelena
"""

from my_functions import add_column_with_surprisal, akaike_for_column
#import warnings
import numpy as np
import pandas as pd
import os

''' Read data '''

# model = 'bertic'
# surprisal = 'surprisal BERTic uni'

#model = 'bert'
#surprisal = 'surprisal BERT uni'

model = 'gpt'
surprisal = 'surprisal GPT'

baseline_model = 'baseline -3'

file_path =  os.path.join('..','podaci','transformer layers parameters', 'datasets', f"{model}.csv")
df = pd.read_csv(file_path)


''' Add model results '''

parameters = [f'CE {j}' for j in range(1, 13)]
for parameter in parameters:
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
    
    
''' Analysis '''

# Grouping by Speaker, Gender, and Emotion, then averaging duration and variability
df_grouped = df.groupby(['speaker', 'speaker gender', 'emotion']).agg({'time': ['mean', 'std']}).reset_index()
df_grouped.columns = [' '.join(col).strip() for col in df_grouped.columns.values]


ll_best_improvement = []
the_best_k = []
ll_k1_improvement = []
x_axis = np.arange(1, 13, 1)

for  _,row in df_grouped.iterrows():
    
    speaker = row['speaker']
    emotion = row['emotion']
    
    filtered_data = df[(df['speaker'] == speaker) & (df['emotion'] == emotion)]
    
    k_list = []
    k_improvements = []
    for parameter in parameters:
        
        difference, _ = akaike_for_column(filtered_data, f"{surprisal} {parameter} model", 'baseline -3')
        k_improvements.append(difference)
        
    max_value = max(k_improvements)  # Find the max value
    max_index = k_improvements.index(max_value)  # Find the index of the max value

    ll_best_improvement.append(max_value)
    the_best_k.append(max_index + 1)
    ll_k1_improvement.append(k_improvements[-1])

df_grouped['optimal k'] = the_best_k
df_grouped['the best LL'] = ll_best_improvement
df_grouped['LL'] = ll_k1_improvement

''' Anlyze results '''

import matplotlib.pyplot as plt
import seaborn as sns

emotion_colour = {
    0: (1, 0, 0, 1),       # red
    1: (0, 0, 1, 1),       # blue
    2: (0, 1, 0, 1),       # green
    3: (1, 0.65, 0, 1),    # orange (RGB ≈ 255,165,0)
    4: (0.5, 0, 0.5, 1)    # purple (RGB ≈ 128,0,128)
}

gender_colour = {
    'f': (1, 0, 0, 1),       # red
    'm': (0, 0, 1, 1)       # blue
}

# Učitaj podatke
df = df_grouped

# Count occurrences of each (emotion, optimal k, speaker gender) combination
df['count'] = df.groupby(['emotion', 'optimal k', 'speaker gender'])['speaker gender'].transform('count')

df_f = df[df['speaker gender'] == 'f']
df_m = df[df['speaker gender'] == 'm']


fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)  # Povećana širina i automatsko prilagođavanje rasporeda
# Prvi subplot
#sns.scatterplot(ax=axes[0], x=df['time mean'], y=df['LL'], hue=df['emotion'], palette='viridis', s=70)  # Povećane tačke
sns.scatterplot(ax=axes[0], x=df['time mean'], y=df['the best LL'], hue=df['emotion'], palette=emotion_colour, s=70)  # Povećane tačke
axes[0].set_xlabel('time mean', fontsize=20)
axes[0].set_ylabel(r'$\Delta$LogLikelihood', fontsize=20)
axes[0].tick_params(axis='both', labelsize=15)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, ['neutral', 'happy', 'sad', 'scared', 'angry'], title='Emotion', fontsize=15, title_fontsize=20, loc='best')
axes[0].axis(ymin=-0.04,ymax=0.125)


# Drugi subplot
sns.scatterplot(ax=axes[1], x=df['time mean'], y=df['LL'], hue=df['emotion'], palette=emotion_colour, s=70)  # Povećane tačke
axes[1].set_xlabel('time mean', fontsize=20)
axes[1].set_ylabel(r'$\Delta$LogLikelihood', fontsize=20)
axes[1].tick_params(axis='both', labelsize=15)
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, ['neutral', 'happy', 'sad', 'scared', 'angry'], title='Emotion', fontsize=15, title_fontsize=20, loc='best')
axes[1].axis(ymin=-0.04,ymax=0.125)

# Dodaj malo šuma kako bi se tačke manje preklapale
df['emotion_jittered'] = df['emotion'] + np.random.uniform(-0.1, 0.1, size=len(df))
df['optimal_k_jittered'] = df['optimal k'] + np.random.uniform(-0.05, 0.05, size=len(df))
# Your original plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['emotion_jittered'], y=df['optimal_k_jittered'], hue=df['speaker gender'], 
                palette=gender_colour, s=80)

plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['neutral', 'happy', 'sad', 'scared', 'angry'], fontsize=20)
plt.xlabel('Emotion', fontsize=20)
plt.ylabel('Optimal Layer', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Get current handles and labels from the plot
handles, labels = plt.gca().get_legend_handles_labels()

# Add both the custom legend and the original legend to the plot
plt.legend(handles=handles, labels=['female', 'male'], 
           title='Speaker Gender', fontsize=15, title_fontsize=20, loc='best')

plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['optimal k'], y=df['the best LL'] - df['LL'], hue=df['speaker gender'], palette=gender_colour, s=100)
# Poboljšanja u čitljivosti
plt.xlabel('Emotion', fontsize=20)
plt.ylabel(r'$\Delta LL_{optimal\ k} - \Delta LL_{k=1}$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Poboljšana legenda
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['female', 'male'], 
           title='Speaker Gender', fontsize=15, title_fontsize=20, loc='best')
plt.show()

