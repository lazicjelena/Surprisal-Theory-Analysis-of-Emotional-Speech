# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:37:25 2024

@author: Jelena
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import numpy as np
import pandas as pd
import warnings
import os
import math 


def inf_k_model(df, k, surprisal):

    surprisal_name = surprisal + ' ' + str(k)
    model_name = surprisal_name + ' model'
    df[surprisal_name] = df[surprisal] ** k
    results_df = pd.DataFrame(columns = df.columns.tolist().append(model_name))

    for fold in df['fold'].unique():

        test_data = df[df['fold'] == fold]

        train_data = df[df['fold'] != fold][['length', 'log probability', surprisal_name]]
        y_train = df[df['fold'] != fold][['time']]
        y_train['time'] = y_train['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))

        
        # reduce outliers
        gaussian_condition = (y_train['time'] - y_train['time'].mean()) / y_train['time'].std() < 3
        train_data = train_data[gaussian_condition]
        y_train = y_train[gaussian_condition]

        model = LinearRegression()
        model.fit(train_data, y_train)

        y_pred = model.predict(test_data[['length', 'log probability', surprisal_name]])

        test_data.loc[:, model_name] = y_pred
        # Concatenate the DataFrames along rows (axis=0)
        test_data = test_data.drop(columns=[surprisal_name])
        results_df = pd.concat([results_df, test_data], axis=0)
        
    return results_df

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

    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2

    return difference, std_ll_2

def calculate_delta_ll(data, surprisal_name, k):

    try:
      delta_ll, std_element = akaike_for_column(data, surprisal_name + ' ' + str(k) + ' model', 'baseline')
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal} at k = {k}")
      return 0, 0
    
    
    
baseline_results_path = os.path.join('..','podaci','results - linear regression', 'baseline_results_data.csv')
baseline_df = pd.read_csv(baseline_results_path)

file_path = os.path.join('..','podaci','training data')
surprisal_column_name = ['Surprisal GPT-2',
                         'Surprisal Yugo', 
                         'Surprisal ngram-3',
                         'Surprisal BERT',
                         'Surprisal BERTic'
                         ]

x_axis = np.arange(0.25, 3, 0.25)

for surprisal in surprisal_column_name:
    
    df_path = os.path.join(file_path, surprisal + '.csv') 
    df = pd.read_csv(df_path)
    #df = df[df['time']!=0]
    #df['time'] = df['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
    df = pd.merge(df, baseline_df, how='left')
    df = df.dropna(subset=['baseline'])
    
    
    results_list = []
    k_list = []
    std_list = []
    emotion_list = []
    gender_list = []
    
    
    warnings.filterwarnings("ignore")
    for i in x_axis:
        
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal)
    
    warnings.resetwarnings()
    
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
            
        for emotion in [0,1,2,3,4]:
                
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            for i in x_axis:
                k = round(i, 2)
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
    results_path = os.path.join('..','podaci','results - linear regression', f"{surprisal}_results.csv")
    results_df.to_csv(results_path, index=False)





