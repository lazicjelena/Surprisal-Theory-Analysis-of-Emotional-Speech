# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:25:34 2025

@author: Jelena
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import numpy as np
import pandas as pd
import math 

def inf_k_model(df, k, surprisal, prosody = 'time', function = 'power'):

    surprisal_name = surprisal + ' ' + str(k)
    model_name = surprisal_name + ' model'
    if function != 'power':
        model_name+= function
    
    if function == 'power':
        df[surprisal_name] = df[surprisal] ** k
    if function == 'linear':
        df[surprisal_name] = df[surprisal] * k
    if function == 'logarithmic':
        df[surprisal_name] = np.log(df[surprisal]) 
    if function == 'exponential':
        df[surprisal_name] = np.exp(df[surprisal])

    
    
    results_df = pd.DataFrame(columns = df.columns.tolist().append(model_name))

    for fold in df['fold'].unique():

        test_data = df[df['fold'] == fold]

        train_data = df[df['fold'] != fold][['length', 'log probability', surprisal_name]]
        y_train = df[df['fold'] != fold][[prosody]]
        y_train[prosody] = y_train[prosody].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
        
        # reduce outliers
        gaussian_condition = (y_train[prosody] - y_train[prosody].mean()) / y_train[prosody].std() < 3
        train_data = train_data[gaussian_condition]
        y_train = y_train[gaussian_condition]

        model = LinearRegression()
        model.fit(train_data, y_train)

        y_pred = model.predict(test_data[['length', 'log probability', surprisal_name]])
        #y_pred = 2**y_pred
        
        test_data.loc[:, model_name] = y_pred
        # Concatenate the DataFrames along rows (axis=0)
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

def akaike_for_column(data, prominence, model_name, baseline_model = 'baseline'):

    _, mean_ll_1, std_ll_1 = calculate_aic(data[prominence], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data[prominence], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2
    std_difference = std_ll_1 - std_ll_2

    return difference, std_difference


def calculate_delta_ll(data, surprisal, k, emotion_data, std_data, prominence = 'time', function = 'power'):

        
    model_name = surprisal + ' ' + str(k) + ' model'
    if function != 'power':
        model_name+= function
        
    try:
      delta_ll, std_list = akaike_for_column(data, prominence,  model_name, 'baseline')
    except:
      delta_ll = [0,0,0,0,0]
      std_list = [1,1,1,1,1]
    for emotion in range(0,5):
      emotion_data[emotion].append(delta_ll[emotion])
      std_data[emotion].append(std_list)

    return

def calculate_delta_ll_old(data, surprisal_name, k, prominence = 'time', function = 'power'):
    
    model_name = surprisal_name + ' ' + str(k) + ' model'
    if function != 'power':
        model_name+= function

    try:
      delta_ll, std_element = akaike_for_column(data, prominence, model_name, 'baseline')
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal_name} at k = {k}")
      return 0, 0

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
                if len(freq) == 1:
                    log_probability_value += freq[column_name].values[0]
                else:
                    log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
            else:
                break
                log_probability_value += 0
                print('error')
                print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s) or word == freq_s['Word'].iloc[-1]:
              list_of_words = []

        log_prob_list.append(log_probability_value)

    return log_prob_list

def add_word_type(data, freq_df, column_name):
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
        log_probability_value = ''
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq_s = freq_df[freq_df['Sentence'] == sentence]
            freq = freq_s[freq_s['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            if not freq.empty:
                log_probability_value += ' '
                log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
            else:
              log_probability_value += ' '
              print('error')
              print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s):
              list_of_words = []

        log_prob_list.append(log_probability_value.strip())

    return log_prob_list

def most_similar_sentence_index(sentence, target_sentence_df):
    # Remove spaces and lowercase the target sentence
    sentence = sentence.lower().replace(' ', '')

    # Function to count common characters
    def common_chars(text):
        text = text.lower().replace(' ', '')
        return len(set(sentence) & set(text))  # Intersection of character sets

    # Apply the function to compute similarity for each sentence
    target_sentence_df['Similarity'] = target_sentence_df['Text'].apply(common_chars)

    # Get the index of the most similar sentence
    most_similar_index = target_sentence_df['Similarity'].idxmax()
    
    return most_similar_index