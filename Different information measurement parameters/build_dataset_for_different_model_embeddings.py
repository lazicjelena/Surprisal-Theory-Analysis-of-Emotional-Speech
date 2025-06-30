# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:12:06 2025

@author: Jelena
"""

import os
import pandas as pd
from my_functions import lookup_features

model = 'roberta'
surprisal_df_name = 'surprisal RoBERTa uni'

model = 'albert'
surprisal_df_name = 'surprisal ALBERT uni'

model = 'bertic'
surprisal_df_name = 'surprisal BERTic uni'

model = 'bert'
surprisal_df_name = 'surprisal BERT uni'

model = 'gpt'
surprisal_df_name = 'surprisal GPT'


embedding_path = os.path.join('..','podaci','transformer layers parameters', 'parameters', f"information_value_{model}.csv")
output_con_path = os.path.join('..','podaci','transformer layers parameters', 'datasets', f"{model}.csv")


''' build data for embeddings research '''

df_path = os.path.join('..','podaci', 'training_data.csv') 
data = pd.read_csv(df_path)
columns = ['word', 'emotion', 'time', 'position', 'target sentence', 'speaker gender',
           'length', 'log probability', 'word type', 'fold', 'speaker', surprisal_df_name]
data = data[columns]
    
embedding_data = pd.read_csv(embedding_path)

ce_parameters = [f'CE {j}' for j in range(1, 13)]
parameters = ce_parameters

# build dataset
for column in parameters:
    
    column_list = lookup_features(data, embedding_data, column)
    data[column] = column_list
    
    data[f"{column} -1"] = data[f"{column}"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -1"]] = pd.NA
    data[f"{column} -2"] = data[f"{column} -1"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -2"]] = pd.NA
    data[f"{column} -3"] = data[f"{column} -2"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -3"]] = pd.NA

for column in ['length', 'log probability', surprisal_df_name]: 
    data[f"{column} -1"] = data[f"{column}"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -1"]] = pd.NA
    data[f"{column} -2"] = data[f"{column} -1"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -2"]] = pd.NA
    data[f"{column} -3"] = data[f"{column} -2"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -3"]] = pd.NA


# Save the concatenated data to a CSV file
data.to_csv(output_con_path, index=False)


''' Add baseline model results '''
from my_functions import add_column, add_column_with_surprisal
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

data = data.replace("nan", np.nan)
data = data[data['time']!=0]

results_df = add_column(data, 3)    
data = pd.merge(data, results_df, how='left')

results_df = add_column_with_surprisal(data, surprisal=surprisal_df_name, k=3)   
data = pd.merge(data, results_df, how='left')


data.to_csv(output_con_path, index=False)
