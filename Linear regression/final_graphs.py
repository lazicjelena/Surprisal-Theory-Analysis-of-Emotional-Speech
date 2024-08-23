# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 05:22:39 2024

@author: Jelena
"""

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import math 
import warnings

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
    
    
    
file_path = output_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)
columns = ['word', 'emotion', 'time', 'speaker gender', 'target sentence', 
           'log probability', 'length', 'surprisal GPT', 'surprisal yugo',
           'surprisal BERT', 'surprisal BERTic', 'surprisal ngram3 alpha4',
           'fold', 'baseline']
df = df[columns]

# make plots english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)
surprisal_list = ['surprisal GPT', 'surprisal yugo', 'surprisal ngram3 alpha4']
surprisal_colour = {'surprisal GPT': (0, 0 , 1, 1), 
                    'surprisal yugo':(1, 0 , 0, 1),
                    'surprisal ngram3 alpha4':(1, 0, 1, 1)}

x_axis = np.arange(0.25, 3, 0.25)

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for i in x_axis:
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal)
warnings.resetwarnings()
    
for gender in ['f', 'm']:
    gender_data = df[df['speaker gender'] == gender]
    
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll(emotion_data, surprisal, k)
                y_axis.append(delta_element)
                y_std.append(std_element)
            
            c = surprisal_colour[surprisal]
            plt.scatter(x_axis, y_axis, s=100, color=c)
            # Add shadows based on the standard deviation of y_axis
            shadow_c = c[:-1] + (0.3,)
            y_std = np.std(y_axis)
            plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)

# Add a common x-axis label
fig.text(0.5, 0.001, 'surprisal power', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['gpt-2','gpt-2 std', 'yugo', 'yugo std','3-gram','3-gram std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


# plot for bidirectional mdoels in english
fig = plt.figure(figsize=(12,8))
emotions = ["neutral", "happy", "sad", "scared", "angry"]
fig.suptitle('Surprisal Power Impact on Spoken Word Duration Prediction', fontsize=30)
surprisal_list = ['surprisal BERT', 'surprisal BERTic']
surprisal_colour = {'surprisal BERT': (0, 0 , 1, 1), 
                    'surprisal BERTic':(1, 0 , 0, 1)}

x_axis = np.arange(0.25, 3, 0.25)

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for i in x_axis:
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal)
warnings.resetwarnings()
    
for gender in ['f', 'm']:
    gender_data = df[df['speaker gender'] == gender]
    
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll(emotion_data, surprisal, k)
                y_axis.append(delta_element)
                y_std.append(std_element)
            
            c = surprisal_colour[surprisal]
            plt.scatter(x_axis, y_axis, s=100, color=c)
            # Add shadows based on the standard deviation of y_axis
            shadow_c = c[:-1] + (0.3,)
            y_std = np.std(y_axis)
            plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)

# Add a common x-axis label
fig.text(0.5, 0.001, 'surprisal power', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['bert','bert std', 'bertic', 'bertic std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()



# make plots serbian
fig = plt.figure(figsize=(12,8))
emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
fig.suptitle('Утицај степена сурприсала на предикцију трајања изговора', fontsize=30)
surprisal_list = ['surprisal GPT', 'surprisal yugo', 'surprisal ngram3 alpha4']
surprisal_colour = {'surprisal GPT': (0, 0 , 1, 1), 
                    'surprisal yugo':(1, 0 , 0, 1),
                    'surprisal ngram3 alpha4':(1, 0, 1, 1)}

x_axis = np.arange(0.25, 3, 0.25)

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for i in x_axis:
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal)
warnings.resetwarnings()
    
for gender in ['f', 'm']:
    gender_data = df[df['speaker gender'] == gender]
    
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll(emotion_data, surprisal, k)
                y_axis.append(delta_element)
                y_std.append(std_element)
            
            c = surprisal_colour[surprisal]
            plt.scatter(x_axis, y_axis, s=100, color=c)
            # Add shadows based on the standard deviation of y_axis
            shadow_c = c[:-1] + (0.3,)
            y_std = np.std(y_axis)
            plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)

# Add a common x-axis label
fig.text(0.5, 0.001, 'степен сурприсала', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['gpt-2','gpt-2 std', 'yugo', 'yugo std','3-gram','3-gram std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


# plot for bidirectional mdoels in serbian
fig = plt.figure(figsize=(12,8))
emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
fig.suptitle('Утицај степена сурприсала на предикцију трајања изговора', fontsize=30)
surprisal_list = ['surprisal BERT', 'surprisal BERTic']
surprisal_colour = {'surprisal BERT': (0, 0 , 1, 1), 
                    'surprisal BERTic':(1, 0 , 0, 1)}

x_axis = np.arange(0.25, 3, 0.25)

warnings.filterwarnings("ignore")
for surprisal in surprisal_list:
    for i in x_axis:
        k = round(i, 2)
        df = inf_k_model(df, k, surprisal)
warnings.resetwarnings()
    
for gender in ['f', 'm']:
    gender_data = df[df['speaker gender'] == gender]
    
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion'] == emotion]
        if gender == 'f':
            plt.subplot(2,5, emotion + 1)
        else:
            plt.subplot(2,5, emotion + 6)
            
        # calculate model predictions
        for surprisal in surprisal_list:
            y_axis = []
            y_std = []
            
            for i in x_axis:
                k = round(i, 2)
                delta_element,  std_element = calculate_delta_ll(emotion_data, surprisal, k)
                y_axis.append(delta_element)
                y_std.append(std_element)
            
            c = surprisal_colour[surprisal]
            plt.scatter(x_axis, y_axis, s=100, color=c)
            # Add shadows based on the standard deviation of y_axis
            shadow_c = c[:-1] + (0.3,)
            y_std = np.std(y_axis)
            plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)

# Add a common x-axis label
fig.text(0.5, 0.001, 'степен сурприсала', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(['bert','bert std', 'bertic', 'bertic std'], fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()