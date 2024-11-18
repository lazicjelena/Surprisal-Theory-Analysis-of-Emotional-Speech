# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:18:41 2024

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

def calculate_delta_ll(data, model_name):

    try:
      delta_ll, std_element = akaike_for_column(data, model_name, "baseline -3")
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {model_name}")
      return 0, 0
    

def add_column_with_surprisal(df, parameter, surprisal, k=3):
    
    columns = df.columns.tolist()
    training_columns = ['length', 'log probability', parameter]
    if surprisal: 
        training_columns.append(surprisal)
    else:
        columns.remove('Surprisal GPT-2')
        for i in range(1,k+1):
            columns.remove(f"Surprisal GPT-2 -{i}")
        
    basic_columns = training_columns.copy()
    for i in range(1,k+1):
        for column in basic_columns:
            training_columns.append(f"{column} -{i}")
            
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
        test_data.loc[:, f"{surprisal} + {parameter} model"] = y_pred
            
        # Concatenate the DataFrames along rows (axis=0)
        results_df = pd.concat([results_df, test_data], axis=0)
        
   
    return results_df.drop_duplicates()


def paired_permutation_test(df, col1, col2, num_permutations=1000):

    # Extract the scores from the two columns
    scores1 = df[col1].values
    scores2 = df[col2].values
    
    # Calculate the observed mean difference
    observed_diff = np.mean(scores1 - scores2)
    #print(observed_diff)
    
    # Initialize a list to store permuted differences
    permuted_diffs = []
    
    # Perform permutations
    for _ in range(num_permutations):
        # Randomly swap each pair of scores
        permuted_scores1, permuted_scores2 = [], []
        for s1, s2 in zip(scores1, scores2):
            if np.random.rand() > 0.5:
                permuted_scores1.append(s1)
                permuted_scores2.append(s2)
            else:
                permuted_scores1.append(s2)
                permuted_scores2.append(s1)
        
        # Calculate the mean difference for this permutation
        permuted_diff = np.mean(np.array(permuted_scores1) - np.array(permuted_scores2))
        permuted_diffs.append(permuted_diff)
    # Calculate p-value
    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

    return p_value    

file_path = os.path.join('..','podaci','information measurements parameters', "data.csv") 
data = pd.read_csv(file_path)
data = data[data['time']!=0]
    
results_list = []
parameter_list = []
emotion_list = []
std_list = []

surprisal = 'Surprisal GPT-2'

#parameters = ['Contextual Entropy']
              
#parameters = ['Context Embedding','Non-context Embedding','Pos-Tag','Orthographic']

parameters = ['AS Context Embedding','AS Non-context Embedding','AS Pos-Tag','AS Orthographic']

# Add surprisal results
df = data.dropna()

if parameters[0] == 'Contextual Entropy':

    results_df = add_column_with_surprisal(data, surprisal, None, 3)
    data = pd.merge(data, results_df, how='left')

    results_df = add_column_with_surprisal(data, 'Contextual Entropy', None, 3)
    data = pd.merge(data, results_df, how='left')

    for emotion in [0,1,2,3,4]:
        emotion_data = df[df['emotion'] == emotion]
                
        delta_element, _ = calculate_delta_ll(emotion_data,  f"None + {surprisal} model")
        std_element = paired_permutation_test(emotion_data, 'baseline -3', f"None + {surprisal} model", 100)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append(surprisal)
    
        delta_element, _ = calculate_delta_ll(emotion_data,  "None + Contextual Entropy model")
        std_element = paired_permutation_test(emotion_data, 'baseline -3', "None + Contextual Entropy model", 100)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append("Contextual Entropy model")
            
    
for parameter in parameters:
    
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
            
    for emotion in [0,1,2,3,4]:
        emotion_data = df[df['emotion'] == emotion]

        delta_element, _ = calculate_delta_ll(emotion_data,  f"{surprisal} + {parameter} model")
        std_element = paired_permutation_test(emotion_data, 'baseline -3', f"{surprisal} + {parameter} model", 100)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append(surprisal + ' ' + parameter)
            
results_df = pd.DataFrame({'y_axis': results_list,
                          'p_value': std_list,
                          'parameter': parameter_list,
                          'emotion': emotion_list})

''' Print results '''
    
parameters = set(results_df['parameter'])

x_axis = [0, 1, 2, 3, 4]

for p in parameters:
    print(p)
    for emotion in x_axis:
            emotion_data = results_df[results_df['emotion'] == emotion]
            value = emotion_data[emotion_data['parameter']==p]['y_axis'].values[0]
            p_value = emotion_data[emotion_data['parameter']==p]['p_value'].values[0]
            print(f"{value:.3f} {p_value:.3f}")

