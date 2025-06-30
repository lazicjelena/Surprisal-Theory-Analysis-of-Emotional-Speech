# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:51:11 2025

@author: Jelena
"""

from my_functions import add_column_with_surprisal, akaike_for_column
#import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

''' Read data '''

model = 'bertic'
surprisal = 'surprisal BERTic uni'

model = 'bert'
surprisal = 'surprisal BERT uni'

# model = 'gpt'
# surprisal = 'surprisal GPT'

baseline_model = 'baseline -3'

file_path =  os.path.join('..','podaci','transformer layers parameters', 'datasets', f"{model}.csv")
df = pd.read_csv(file_path)
df = df.dropna()

''' Add model results '''

parameters = [f'CE {j}' for j in range(1, 13)]

for parameter in parameters:
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
    
columns = ['word', 'emotion', 'time', 'position', 'target sentence',
       'speaker gender', 'length', 'log probability', 'word type', 'fold',
       'speaker', surprisal, 'baseline -3']

for i in range(1,13):
    columns.append(f"{surprisal} CE {i} model")

df = df[columns]

for parameter in parameters:
    column_name = f"{surprisal} {parameter} model"
    df[column_name] = np.exp(df[column_name]) - df['time']


''' Anlyze results '''
for i in range(1,13):
    
    # Plotovanje
    plt.figure(figsize=(10, 6))
    
    plt.hist(df[f"{surprisal} CE 12 model"].dropna(), bins=2000, color='skyblue', alpha=0.7, density=True, edgecolor='black', label="layer 12")
    plt.hist(df[f"{surprisal} CE {i} model"].dropna(), bins=2000, color='lightpink', alpha=0.7, density=True, edgecolor='black', label=f"layer {i}")
    
    plt.xlabel("values", fontsize = 20)
    plt.ylabel("frequency", fontsize = 20)
    plt.title(f"Layer {i}", fontsize=20)
    plt.legend(fontsize = 15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim([-2, 2])
    #plt.tight_layout()
    plt.show()

    
