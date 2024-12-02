# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:36:17 2024

@author: Jelena
"""
import os
import pandas as pd
from my_functions import lookup_features

inform_value = False

if inform_value:
    embedding_path = os.path.join('..','podaci','information measurements parameters', "embedding_information_value.csv")
    output_con_path = os.path.join('..','podaci','information measurements parameters', "context_embedding_data.csv")
    output_nc_path = os.path.join('..','podaci','information measurements parameters', "non_context_embedding_data.csv")
else:
    embedding_path = os.path.join('..','podaci','information measurements parameters', "embedding_adjusted_surprisal.csv")
    output_con_path = os.path.join('..','podaci','information measurements parameters', "context_embedding_data_surprisal.csv")
    output_nc_path = os.path.join('..','podaci','information measurements parameters', "non_context_embedding_data_surprisal.csv")


''' build data for embeddings research '''

df_path = os.path.join('..','podaci', 'training data', "Surprisal GPT-2.csv") 
data = pd.read_csv(df_path)
data = data.drop(columns=['speaker'])
    
embedding_data = pd.read_csv(embedding_path)

ce_parameters = [f'CE {j}' for j in range(1, 13)]
nce_parameters = [f'NCE {j}' for j in range(1, 13)]
parameters = ce_parameters + nce_parameters

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

for column in ['length', 'log probability', 'Surprisal GPT-2']: 
    data[f"{column} -1"] = data[f"{column}"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -1"]] = pd.NA
    data[f"{column} -2"] = data[f"{column} -1"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -2"]] = pd.NA
    data[f"{column} -3"] = data[f"{column} -2"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -3"]] = pd.NA

# save non context data info    
pom = data.drop(columns=ce_parameters)
for i in range(1,4):
    pom = pom.drop(columns=[f'CE {j} -{i}' for j in range(1, 13)])
   
# Save the concatenated data to a CSV file
pom.to_csv(output_nc_path, index=False)

# save context data info
pom = data.drop(columns=nce_parameters)
for i in range(1,4):
    pom = pom.drop(columns=[f'NCE {j} -{i}' for j in range(1, 13)])
   
# Save the concatenated data to a CSV file
pom.to_csv(output_con_path, index=False)