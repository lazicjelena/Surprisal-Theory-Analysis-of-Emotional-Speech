# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:34:49 2025

@author: Jelena
"""


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import math

prosody = 'energy'
file_path =  os.path.join('..','podaci', f"{prosody}_data.csv")
df = pd.read_csv(file_path)

# add length and log probability
df['length'] = df['word'].apply(lambda x: len(x))
df = df[df['surprisal BERT']!=0]

# add log probabilities
f_path = os.path.join('..','podaci', 'wordlist_frequencies.csv') 
freq_df = pd.read_csv(f_path)

log_probability_list = []
for _,row in df.iterrows():
    
    words = row['word'].lower().strip()
    lp_value = 0
    for word in words.split(' '):
        if word =='dogovrili':
            word ='dogovorili'
        if word =='pet':
            word = '5'
        lp_value += freq_df[freq_df['Word'] == word]['Log Probability'].iloc[0]
        
    log_probability_list.append(lp_value)
        
df['log probability'] = log_probability_list

# add baseline results
mse_list = []
results_df = pd.DataFrame(columns = df.columns.tolist().append('baseline'))

df = df.dropna(subset = [prosody])

for fold in df['fold'].unique():

    test_data = df[df['fold'] == fold]
    y_test = df[df['fold'] == fold][[prosody]]
    #y_test[prosody] = y_test[prosody].apply(lambda x: math.log2(x) if x > 0 else float('nan'))

    train_data = df[df['fold'] != fold]
    y_train = df[df['fold'] != fold][[prosody]]
    y_train[prosody] = y_train[prosody].apply(lambda x: math.log2(x) if x > 0 else float('nan'))


    # reduce outliers
    gaussian_condition = (y_train[prosody] - y_train[prosody].mean()) / y_train[prosody].std() < 3
    train_data = train_data[gaussian_condition]
    y_train = y_train[gaussian_condition]

    model = LinearRegression()
    model.fit(train_data[['length', 'log probability']], y_train)

    y_pred = model.predict(test_data[['length', 'log probability']])
    #y_pred = 2**y_pred

    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    test_data.loc[:, 'baseline'] = y_pred
    # Concatenate the DataFrames along rows (axis=0)
    results_df = pd.concat([results_df, test_data], axis=0)


# Calculate the average of mse_list
average_mse = sum(mse_list) / len(mse_list)

for f, mse in zip(df['fold'].unique(),mse_list):
  print(f"Fold: {f}  MSE: {mse}")
print(f"Average mse over folds: {average_mse}")

results_df

results_df.to_csv(file_path, index=False)