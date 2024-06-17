# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:25:29 2024
@author: Jelena
lazic.jelenaa@gmail.com

Ova skripta prikazuje rezultate korelacije parametara wavelet transform i 
surprisala razlicitih model. Rezultati se koriste u radu.
"""

import pandas as pd
import os

data_path = os.path.join('..','podaci', 'prominence_data.csv') 
data = pd.read_csv(data_path)

# Selecting the columns for correlation
columns_of_interest = ['surprisal BERT', 'surprisal BERTic', 'surprisal ngram3 alpha4', 'surprisal GPT', 'surprisal yugo']
emotions = [0, 1, 2, 3, 4]

data = data[data['gender']=='m']
for column in columns_of_interest:
    print(f"For model {column} correlation values are following: ")
    for emotion in emotions:
        df = data[data['emotion']==emotion]
        correlation_value = df[column].corr(df['prominence'])
        #print(f"{emotion} {correlation_value}")
        print(f"{correlation_value:.4f}")
        
