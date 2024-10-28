# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:37:25 2024

@author: Jelena
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import numpy as np
import pandas as pd
import os
import math 

def calculate_log_Likelihood(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.logpdf(data, loc=mean, scale=std_dev)

# Calculate AIC for models with different numbers of parameters
def calculate_aic(real_values, results, k):
    residuals = np.array(real_values) - np.array(results)
    log_likelihood = calculate_log_Likelihood(residuals)
    aic = 2 * k - 2 * log_likelihood
    return aic, np.mean(log_likelihood), np.std(log_likelihood)

def akaike_for_column(data, model_name, baseline_model = 'baseline'):
    
    data = data.dropna(subset=[model_name, baseline_model])
    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2

    return difference, std_ll_2

def calculate_delta_ll(data, surprisal_name, k):

    try:
      delta_ll, std_element = akaike_for_column(data, f"{surprisal_name} -{k} model", f"baseline -{k}")
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal} at k = {k}")
      return 0, 0
    

def add_column_with_surprisal(df, surprisal, k=0):
    
    columns = df.columns.tolist()
    training_columns = ['length', 'log probability', surprisal]
    if k:
        for i in range(1,k+1):
            training_columns.append(f"length -{i}")
            training_columns.append(f"log probability -{i}")
            training_columns.append(f"{surprisal} -{i}")
            
    # Assuming 'columns' and 'training_columns' are your lists
    columns.extend([col for col in training_columns if col not in columns])
    results_df = pd.DataFrame(columns = columns)
        
    df = df[(~df[training_columns].isna()).all(axis=1)]

    for fold in df['fold'].unique():

        test_data = df[df['fold'] == fold]
        y_test = df[df['fold'] == fold][['time']]
        
        train_data = df[df['fold'] != fold]
        y_train = df[df['fold'] != fold][['time']]
        y_train['time'] = y_train['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
        
        # reduce outliers
        gaussian_condition = (y_train['time'] - y_train['time'].mean()) / y_train['time'].std() < 3
        train_data = train_data[gaussian_condition]
        y_train = y_train[gaussian_condition]
        
        model = LinearRegression()
        model.fit(train_data[training_columns], y_train)
        
        y_pred = model.predict(test_data[training_columns])
        test_data.loc[:, f"{surprisal} -{k} model"] = y_pred
            
        # Concatenate the DataFrames along rows (axis=0)
        results_df = pd.concat([results_df, test_data], axis=0)
        
   
    return results_df.drop_duplicates()
    
    columns = df.columns.tolist()
    training_columns = ['length', 'log probability']
    if k:
        for i in range(1,k+1):
            training_columns.append(f"length -{i}")
            training_columns.append(f"log probability -{i}")
            
    # Assuming 'columns' and 'training_columns' are your lists
    columns.extend([col for col in training_columns if col not in columns])
    results_df = pd.DataFrame(columns = columns)
        
    df = df[(~df[training_columns].isna()).all(axis=1)]

    for fold in df['fold'].unique():

        test_data = df[df['fold'] == fold]
        y_test = df[df['fold'] == fold][['time']]
        
        train_data = df[df['fold'] != fold]
        y_train = df[df['fold'] != fold][['time']]
        y_train['time'] = y_train['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
        
        # reduce outliers
        gaussian_condition = (y_train['time'] - y_train['time'].mean()) / y_train['time'].std() < 3
        train_data = train_data[gaussian_condition]
        y_train = y_train[gaussian_condition]
        
        model = LinearRegression()
        model.fit(train_data[training_columns], y_train)
        
        y_pred = model.predict(test_data[training_columns])
        test_data.loc[:, f"baseline {k}"] = y_pred
            
        # Concatenate the DataFrames along rows (axis=0)
        results_df = pd.concat([results_df, test_data], axis=0)
        
        y_test['time'] = y_test['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
    
    return results_df.drop_duplicates()
    
baseline_results_path = os.path.join('..','podaci','split-over results', 'baseline_results_data.csv') 
baseline_df = pd.read_csv(baseline_results_path)

file_path = os.path.join('..','podaci','split-over data')
surprisal_column_name = ['Surprisal GPT-2',
                         'Surprisal Yugo', 
                         'Surprisal ngram-3',
                         'Surprisal BERT',
                         'Surprisal BERTic'
                         ]

k_values = [0, 1, 2, 3, 4]

for surprisal in surprisal_column_name:
    
    df_path = os.path.join(file_path, surprisal + '.csv') 
    df = pd.read_csv(df_path)
    df = df[df['time']!=0]
    df = pd.merge(df, baseline_df, how='left').drop_duplicates()
    
    results_list = []
    k_list = []
    emotion_list = []
    gender_list = []
    std_list = []
    
    for k in k_values:
        results_df = add_column_with_surprisal(df, surprisal, k)
        df = pd.merge(df, results_df, how='left')

        for gender in ['f', 'm']:
            gender_data = df[df['speaker gender'] == gender]
            
            for emotion in [0,1,2,3,4]:   
                emotion_data = gender_data[gender_data['emotion'] == emotion]
            
                delta_element, std_element = calculate_delta_ll(emotion_data, surprisal, k)
                
                gender_list.append(gender)
                results_list.append(delta_element)
                std_list.append(std_element)
                emotion_list.append(emotion)
                k_list.append(k)
            
    data = {
        'y_axis': results_list,
        'y_std': std_list,
        'k': k_list,
        'emotion': emotion_list,
        'speaker gender': gender_list
    }
    
    results_df = pd.DataFrame(data)
    results_path = os.path.join('..','podaci','split-over results', f"{surprisal}_results.csv")
    results_df.to_csv(results_path, index=False)





