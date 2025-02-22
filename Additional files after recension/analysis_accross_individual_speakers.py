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

columns_to_remove = ['surprisal ngram2 alpha4', 'surprisal ngram3 alpha4', 'surprisal ngram4 alpha4',
                     'surprisal ngram5 alpha4', 'surprisal ngram2 alpha20', 'surprisal ngram3 alpha20', 
                     'surprisal ngram4 alpha20', 'surprisal ngram5 alpha20','surprisal BERT', 
                     'surprisal BERTic', 'surprisal GPT3', 'surprisal yugo']

df = df.drop(columns = columns_to_remove)

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
        
        difference = akaike_for_column(filtered_data,  f"surprisal GPT {str(k)} model", 'baseline')
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

# Show the plot
plt.show()

sns.scatterplot(x=df['time mean'], y=df['LL'], hue=df['emotion'], palette='viridis')
plt.xlabel('time mean')
plt.ylabel('LL')
plt.show()


sns.scatterplot(x=df['time mean'], y=df['LL'], hue=df['speaker gender'], palette='viridis')
plt.xlabel('time mean')
plt.ylabel('LL')
plt.show()



# Add slight random noise to avoid overlap (adjust scale as needed)
df['emotion_jittered'] = df['emotion'] + np.random.uniform(-0.1, 0.1, size=len(df))
df['optimal_k_jittered'] = df['optimal k'] + np.random.uniform(-0.05, 0.05, size=len(df))
sns.scatterplot(x=df['emotion_jittered'], y=df['optimal_k_jittered'], hue=df['speaker gender'], 
                palette='viridis')
plt.xlabel('Emotion')
plt.ylabel('Optimal k')
plt.title('Scatter Plot with Jitter to Reduce Overlaps')
plt.legend(title='Speaker Gender')
#plt.legend(title='Speaker Gender', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()


sns.scatterplot(x=df['optimal k'], y=df['the best LL'] - df['LL'], hue=df['speaker gender'], palette='viridis')
plt.xlabel('optimal k')
plt.ylabel('the best LL - LL')
plt.show()