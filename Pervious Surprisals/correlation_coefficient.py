# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 03:42:07 2024

@author: Jelena

U ovoj skripti proucavamo funkcije koje se mogu koristiti za racunanje koeficijenta
korelacije izmedju vise promjenjivih.
"""

# pip install scikit-learn

import numpy as np
import pandas as pd
import os
import scipy.stats

def lookup_features(data, surprisal_df, column_name):
    surprisal_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        words = row['word'].split(' ')
        sentence = row['target sentence']
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
          
        surprisal_value = 0
        for word in words:
            # Filter freq_df based on the 'Word' column
            surprisal_s = surprisal_df[surprisal_df['target sentence'] == sentence]
            surprisal_w = surprisal_s[surprisal_s['word'] == word]

            # Extract the 'Surprisal' value for the filtered word
            if len(surprisal_w)>0:
                surprisal_value += surprisal_w[column_name].values[0 + list_of_words.count(word)]
            else:
                surprisal_value += 0
               # print('error')
               # print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(surprisal_s):
              list_of_words = []

        surprisal_list.append(surprisal_value)

    return surprisal_list

# read surprisal data
surprisal_path = os.path.join('..','podaci', 'correlation data', "former_" + 'word_surprisals_gpt2.csv') 
surprisal_df = pd.read_csv(surprisal_path)

# read prominence data
prominence_path = os.path.join('..','podaci', 'correlation data','prominence_data.csv') 
prominence_df = pd.read_csv(prominence_path)

# conjoint dataframe
column_name = 'Surprisal GPT-2'
prominence_df[column_name] = lookup_features(prominence_df, surprisal_df, column_name)

x = prominence_df[column_name].values
y = prominence_df['time'].values

np_corrcoef = np.corrcoef(x, y)
print(f"Numpy: {np_corrcoef}")

pearson_corrcoef = scipy.stats.pearsonr(x, y) 
print(f"Pearson: {scipy.stats.pearsonr(x, y)}")
    
spearman_corrcoef = scipy.stats.spearmanr(x, y) 
print(f"Spearman: {spearman_corrcoef}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create and fit the regression model
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

# Predict Y values based on the model
y_pred = model.predict(x.reshape(-1, 1))

# Calculate R-squared
r_squared = r2_score(y, y_pred)

# Calculate the multiple correlation coefficient
lr_corrcoef = r_squared ** 0.5
print(f"LR: {lr_corrcoef}")


# for multiple x
column_name = 'Surprisal GPT-2 k=1'
prominence_df[column_name] = lookup_features(prominence_df, surprisal_df, column_name)
prominence_df = prominence_df.dropna(subset=[column_name])
x = prominence_df[['Surprisal GPT-2', 'Surprisal GPT-2 k=1']]
y = prominence_df['time'].values

# Create and fit the regression model
model = LinearRegression()
model.fit(x, y)

# Predict Y values based on the model
y_pred = model.predict(x)

# Calculate R-squared
r_squared = r2_score(y, y_pred)

# Calculate the multiple correlation coefficient
lr_corrcoef = r_squared ** 0.5
print(f"LR: {lr_corrcoef}")

# for single former surprisal:
column_name = 'Surprisal GPT-2 k=2'
prominence_df[column_name] = lookup_features(prominence_df, surprisal_df, column_name)
prominence_df = prominence_df.dropna(subset=[column_name])
x = prominence_df[column_name].values
y = prominence_df['time'].values
# Create and fit the regression model
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

# Predict Y values based on the model
y_pred = model.predict(x.reshape(-1, 1))

# Calculate R-squared
r_squared = r2_score(y, y_pred)

# Calculate the multiple correlation coefficient
lr_corrcoef = r_squared ** 0.5
print(f"LR: {lr_corrcoef}")
