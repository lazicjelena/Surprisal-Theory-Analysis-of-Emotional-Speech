# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:17:57 2024

@author: Jelena
"""
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import numpy as np
import math 
import pandas as pd

def lookup_features(data, freq_df, column_name):
    log_prob_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        words = row['word'].split(' ')
        sentence = row['target sentence']
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
        #print(index)
        log_probability_value = 0
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq_s = freq_df[freq_df['Sentence'] == sentence]
            freq = freq_s[freq_s['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            if not freq.empty:
                try:
                    log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
                except:
                    log_probability_value += freq[column_name].values[0]
            else:
                log_probability_value += 0
                print('error')
                print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s):
              list_of_words = []

        log_prob_list.append(log_probability_value)

    return log_prob_list

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

def calculate_delta_ll(data, model_name, baseline = "baseline -3"):

    try:
      delta_ll, std_element = akaike_for_column(data, model_name, baseline)
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {model_name}")
      return 0, 0
    
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

def add_column_with_surprisal(df, parameter='', surprisal='', k=3):
    '''
    Parameters
    ----------
    df : dataframe
        Training data.
    parameter : str
        Column name for additional column to take as input to LR model.
    surprisal : str
        Surprisal column name.
    k : int
        Split over effect, optional. The default is 3.

    Returns
    -------
    results_df:
        initial data with additional column for LR model results for prediction with parameter.

    '''
    
    columns = df.columns.tolist()
    
    if parameter != '':
        training_columns = ['length', 'log probability', parameter]

    else:
        training_columns = ['length', 'log probability']
    
    if surprisal != '': 
        training_columns.append(surprisal)
    # else:
    #     columns.remove(surprisal)
    #     for i in range(1,k+1):
    #         columns.remove(f"{surprisal} -{i}")
            
    # create column names
    if surprisal != '': 
        result_column_name = surprisal + ' '
    else:
        result_column_name = ''
    if parameter != '':
        result_column_name += parameter + ' '
        
    result_column_name += 'model'
        
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
        #y_test = df[df['fold'] == fold][['time']]
        
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
        test_data.loc[:, result_column_name] = y_pred
            
        # Concatenate the DataFrames along rows (axis=0)
        results_df = pd.concat([results_df, test_data], axis=0)
        
   
    return results_df.drop_duplicates()


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
        #y_test = df[df['fold'] == fold][['time']]
        
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
