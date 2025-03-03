# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:05:22 2025

@author: Jelena
"""

from my_functions import inf_k_model, akaike_for_column
import numpy as np
import pandas as pd
import os

file_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)

#columns_to_remove = ['surprisal GPT']
#df = df.drop(columns = columns_to_remove)

import warnings
# Filter out SettingWithCopyWarning
warnings.filterwarnings("ignore")
x_axis = np.arange(0.25, 3, 0.25)

for i in x_axis:
  k = round(i, 2)
  df = inf_k_model(df, k, 'surprisal GPT')

# Reset warnings to default behavior (optional)
warnings.resetwarnings()

# Grouping by Speaker, Gender, and Emotion, then averaging duration and variability
df_grouped = df.groupby(['speaker', 'speaker gender', 'emotion']).agg({'time': ['mean', 'std']}).reset_index()
df_grouped.columns = [' '.join(col).strip() for col in df_grouped.columns.values]


ll_best_improvement = []
the_best_k = []
ll_k1_improvement = []

for  _,row in df_grouped.iterrows():
    
    speaker = row['speaker']
    emotion = row['emotion']
    
    filtered_data = df[(df['speaker'] == speaker) & (df['emotion'] == emotion)]
    
    k_list = []
    k_improvements = []
    for i in x_axis:
        k = round(i, 2)
        k_list.append(k)
        
        difference, _ = akaike_for_column(filtered_data, 'time',  f"surprisal GPT {str(k)} model", 'baseline')
        k_improvements.append(difference)
        
    max_value = max(k_improvements)  # Find the max value
    max_index = k_improvements.index(max_value)  # Find the index of the max value

    ll_best_improvement.append(max_value)
    the_best_k.append(k_list[max_index])
    ll_k1_improvement.append(k_improvements[3])

df_grouped['optimal k'] = the_best_k
df_grouped['the best LL'] = ll_best_improvement
df_grouped['LL'] = ll_k1_improvement


''' Anlyze results '''

import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
df = df_grouped

# Count occurrences of each (emotion, optimal k, speaker gender) combination
df['count'] = df.groupby(['emotion', 'optimal k', 'speaker gender'])['speaker gender'].transform('count')

df_f = df[df['speaker gender'] == 'f']
df_m = df[df['speaker gender'] == 'm']


fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)  # Povećana širina i automatsko prilagođavanje rasporeda
# Prvi subplot
sns.scatterplot(ax=axes[0], x=df['time mean'], y=df['LL'], hue=df['emotion'], palette='viridis', s=70)  # Povećane tačke
axes[0].set_xlabel('time mean', fontsize=20)
axes[0].set_ylabel(r'$\Delta$LogLikelihood', fontsize=20)
axes[0].tick_params(axis='both', labelsize=15)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, ['neutral', 'happy', 'sad', 'scared', 'angry'], title='Emotion', fontsize=15, title_fontsize=20, loc='best')


# Drugi subplot
sns.scatterplot(ax=axes[1], x=df['time mean'], y=df['LL'], hue=df['speaker gender'], palette='viridis', s=70)  # Povećane tačke
axes[1].set_xlabel('time mean', fontsize=20)
axes[1].set_ylabel(r'$\Delta$LogLikelihood', fontsize=20)
axes[1].tick_params(axis='both', labelsize=15)
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, ['female', 'male'], title='Speaker Gender', fontsize=15, title_fontsize=20, loc='best')
plt.show()



# Dodaj malo šuma kako bi se tačke manje preklapale
df['emotion_jittered'] = df['emotion'] + np.random.uniform(-0.1, 0.1, size=len(df))
df['optimal_k_jittered'] = df['optimal k'] + np.random.uniform(-0.05, 0.05, size=len(df))
# Your original plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['emotion_jittered'], y=df['optimal_k_jittered'], hue=df['speaker gender'], 
                palette='viridis', s=80)

plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['neutral', 'happy', 'sad', 'scared', 'angry'], fontsize=20)
plt.xlabel('Emotion', fontsize=20)
plt.ylabel('Optimal k', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Get current handles and labels from the plot
handles, labels = plt.gca().get_legend_handles_labels()

# Add both the custom legend and the original legend to the plot
plt.legend(handles=handles, labels=['female', 'male'], 
           title='Speaker Gender', fontsize=15, title_fontsize=20, loc='best')

plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['optimal k'], y=df['the best LL'] - df['LL'], hue=df['speaker gender'], palette='viridis', s=100)
# Poboljšanja u čitljivosti
plt.xlabel('Optimal k', fontsize=20)
plt.ylabel(r'$\Delta LL_{optimal\ k} - \Delta LL_{k=1}$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Poboljšana legenda
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['female', 'male'], 
           title='Speaker Gender', fontsize=15, title_fontsize=20, loc='best')
plt.show()