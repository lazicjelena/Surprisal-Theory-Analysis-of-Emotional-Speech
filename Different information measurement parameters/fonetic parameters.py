# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:26:07 2024

@author: Jelena
"""


import pandas as pd
import os

df_path = os.path.join('..','podaci', 'training data', "Surprisal GPT-2.csv") 
data = pd.read_csv(df_path)
data = data.drop(columns=['speaker'])
                         
# Add new column for fonetci features
ff_path = os.path.join('..','podaci', 'information measurements parameters', "fonetic_features1.csv") 
ff_data = pd.read_csv(ff_path)
ff_data = ff_data.drop('sentence', axis=1)
ff_data = ff_data.drop_duplicates()

data = data.merge(ff_data, how='left')

columns_list = ['Surprisal GPT-2',
                'length',
                'log probability'
                ]

for column in columns_list:
    # Add new columns for the length and log probability of the previous word
    data[f"{column} -1"] = data[f"{column}"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -1"]] = pd.NA
    data[f"{column} -2"] = data[f"{column} -1"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -2"]] = pd.NA
    data[f"{column} -3"] = data[f"{column} -2"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -3"]] = pd.NA
    

column_names = data.columns.tolist()
column_names = [x for x in column_names if x != 'time'] 
# Group by all columns except 'time' and calculate the mean for 'time'
#data = data.groupby(column_names, as_index=False).agg({'time': 'mean'})                        

''' Make baseline model '''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math
import os
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def add_column(df, k=0):
    
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

    mse_list = []
    
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
        test_data.loc[:, f"baseline -{k}"] = y_pred
            
        # Concatenate the DataFrames along rows (axis=0)
        results_df = pd.concat([results_df, test_data], axis=0)
        
        y_test['time'] = y_test['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        
    # Calculate the average of mse_list
    average_mse = sum(mse_list) / len(mse_list)
    print(f"Average mse over folds for k={k}: {average_mse}")
    
    return results_df.drop_duplicates()


data = data.replace("nan", np.nan)
data = data[data['time']!=0]

results_df = add_column(data, 3)    
data = pd.merge(data, results_df, how='left')

''' Add fonetic parameters '''
from scipy.stats import norm

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

    return difference, std_ll_1

def calculate_delta_ll(data, model_name):

    try:
      delta_ll, std_element = akaike_for_column(data, model_name, "baseline -3")
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {model_name}")
      return 0, 0
    

def fonetic_model(df, fonem_list):
    
    columns = ['length', 'log probability', 'Surprisal GPT-2']
    for i in range(1,4):
        columns.append(f"length -{i}")
        columns.append(f"log probability -{i}")   
    
    columns = columns + fonem_list
    df = df[(~df[columns].isna()).all(axis=1)]
    
    result_df_columns = df.columns.tolist() 
    result_df_columns.extend([col for col in columns if col not in result_df_columns])
    results_df = pd.DataFrame(columns = result_df_columns)

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
        model.fit(train_data[columns], y_train)
        
        y_pred = model.predict(test_data[columns])
        test_data.loc[:, "fonetic model"] = y_pred
            
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
    p_value = np.mean(np.abs(permuted_diffs) > np.abs(observed_diff))

    return p_value

results_list = []
parameter_list = []
emotion_list = []
std_list = []

#fonem_types = ['vokali','sonanti', 'konsonanti']
               
#fonem_types = ['vokali', 'alveolarni', 'palatalni', 'bilabijalni',
#               'labio_dentalni', 'labialni', 'zubni', 'palatalni', 'zadnjonepcani']

#fonem_types = ['zvucni', 'bezvucni']

fonem_types = ['vokali', 'alveolarni', 'palatalni', 'bilabijalni', 'labio_dentalni',
               'labialni', 'zubni', 'palatalni', 'zadnjonepcani', 'zvucni', 'bezvucni']


pom_df = fonetic_model(data, fonem_types)
data = pd.merge(data, pom_df, how='left')
df = data.dropna()
    
for emotion in [0,1,2,3,4]:
    emotion_data = df[df['emotion'] == emotion]
            
    delta_element, _ = calculate_delta_ll(emotion_data, "fonetic model")
    std_element = paired_permutation_test(emotion_data, 'baseline -3', 'fonetic model')
    results_list.append(delta_element)
    std_list.append(std_element)
    emotion_list.append(emotion)
    parameter_list.append('Fonetic')
            
            
results_df = pd.DataFrame({'y_axis': results_list,
                           'p_value': std_list,
                           'parameter': parameter_list,
                           'emotion': emotion_list})

''' Print results '''

parameters = ['Fonetic']
x_axis = [0, 1, 2, 3, 4]


for p in parameters:
    for emotion in x_axis:
            emotion_data = results_df[results_df['emotion'] == emotion]
            value = emotion_data[emotion_data['parameter']==p]['y_axis'].values[0]
            p_value = emotion_data[emotion_data['parameter']==p]['p_value'].values[0]
            print(f"{value:.3f} {p_value:.3f}")
      
        
''' Plot feature distribution '''
import matplotlib.pyplot as plt

# Assuming df is your dataframe with the specified columns
# Summing the values of each column
column_sums = df[fonem_types].sum()

# Plotting the results in a bar chart
plt.figure(figsize=(10, 6))
column_sums.plot(kind='bar', color='blue')
plt.title('Sum of Each Phonetic Feature', fontsize = 25)
plt.xlabel('Phonetic Feature', fontsize = 20)
plt.ylabel('Sum', fontsize = 20)
plt.xticks(rotation=45, fontsize = 15)
plt.show()

