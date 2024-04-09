# -*- coding: utf-8 -*-
"""results.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta samo plotuje sve rezultate onako kako su prikazani u radu.
"""
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

def akaike_for_column(column_name, model_name, baseline_model = 'baseline'):
    difference = []

    for gender in df[column_name].unique():

        data = df[df[column_name]==gender]

        _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
        _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
        difference.append(mean_ll_1-mean_ll_2)

    return difference, std_ll_2

def calculate_delta_ll(surprisal, k, emotion_data, std_data):

    try:
      delta_ll,std_list = akaike_for_column('emotion', surprisal + ' ' + str(k) + ' model', 'baseline')
    except:
      print(f"{surprisal} at k = {k}")
      delta_ll = [0,0,0,0,0]
      std_list = [1,1,1,1,1]
    for emotion in range(0,5):
      emotion_data[emotion].append(delta_ll[emotion])
      std_data[emotion].append(std_list)

    return


file_path = output_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)
df = df[df['speaker gender']=='m']

import warnings
# Filter out SettingWithCopyWarning
warnings.filterwarnings("ignore")
surprisal_gpt_2 = 'surprisal GPT'
surprisal_gpt_3 = 'surprisal GPT3'
surprisal_bert = 'surprisal BERT'
surprisal_bertic = 'surprisal BERTic'
surprisal_ngram_2_alpha4 = 'surprisal ngram2 alpha4'
surprisal_ngram_3_alpha4 = 'surprisal ngram3 alpha4'
surprisal_ngram_4_alpha4 = 'surprisal ngram4 alpha4'
surprisal_ngram_5_alpha4 = 'surprisal ngram5 alpha4'
surprisal_ngram_2_alpha20 = 'surprisal ngram2 alpha20'
surprisal_ngram_3_alpha20 = 'surprisal ngram3 alpha20'
surprisal_ngram_4_alpha20 = 'surprisal ngram4 alpha20'
surprisal_ngram_5_alpha20 = 'surprisal ngram5 alpha20'
surprisal_yugo = 'surprisal yugo'

x_axis = np.arange(0.25, 3, 0.25)

for i in x_axis:
  k = round(i, 2)
  df = inf_k_model(df, k, surprisal_gpt_2)
  df = inf_k_model(df, k, surprisal_gpt_3)
  df = inf_k_model(df, k, surprisal_bert)
  df = inf_k_model(df, k, surprisal_bertic)
  df = inf_k_model(df, k, surprisal_ngram_2_alpha4)
  df = inf_k_model(df, k, surprisal_ngram_3_alpha4)
  df = inf_k_model(df, k, surprisal_ngram_4_alpha4)
  df = inf_k_model(df, k, surprisal_ngram_5_alpha4)
  df = inf_k_model(df, k, surprisal_ngram_2_alpha20)
  df = inf_k_model(df, k, surprisal_ngram_3_alpha20)
  df = inf_k_model(df, k, surprisal_ngram_4_alpha20)
  df = inf_k_model(df, k, surprisal_ngram_5_alpha20)
  df = inf_k_model(df, k, surprisal_yugo)

# Reset warnings to default behavior (optional)
warnings.resetwarnings()

# Initialize an empty dictionary to store emotion-wise data
emotion_data_gpt_2 = { 0: [], 1: [], 2: [], 3: [], 4: []}
gpt_std_2 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_gpt_3 = { 0: [], 1: [], 2: [], 3: [], 4: []}
gpt_std_3 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_bert = { 0: [], 1: [], 2: [], 3: [], 4: []}
bert_std = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_bertic = { 0: [], 1: [], 2: [], 3: [], 4: []}
bertic_std = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_yugo = { 0: [], 1: [], 2: [], 3: [], 4: []}
yugo_std = { 0: [], 1: [], 2: [], 3: [], 4: []}

emotion_data_ngram_2_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_2_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_ngram_3_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_3_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_ngram_4_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_4_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_ngram_5_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_5_alpha4 = { 0: [], 1: [], 2: [], 3: [], 4: []}

emotion_data_ngram_2_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_2_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_ngram_3_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_3_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_ngram_4_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_4_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
emotion_data_ngram_5_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}
ngram_std_5_alpha20 = { 0: [], 1: [], 2: [], 3: [], 4: []}

for i in x_axis:
    k = round(i, 2)
    calculate_delta_ll(surprisal_gpt_2, k, emotion_data_gpt_2, gpt_std_2)
    calculate_delta_ll(surprisal_gpt_3, k, emotion_data_gpt_3, gpt_std_3)
    calculate_delta_ll(surprisal_bert, k, emotion_data_bert, bert_std)
    calculate_delta_ll(surprisal_bertic, k, emotion_data_bertic, bertic_std)
    calculate_delta_ll(surprisal_yugo, k, emotion_data_yugo, yugo_std)
    calculate_delta_ll(surprisal_ngram_2_alpha4, k, emotion_data_ngram_2_alpha4, ngram_std_2_alpha4)
    calculate_delta_ll(surprisal_ngram_3_alpha4, k, emotion_data_ngram_3_alpha4, ngram_std_3_alpha4)
    calculate_delta_ll(surprisal_ngram_4_alpha4, k, emotion_data_ngram_4_alpha4, ngram_std_4_alpha4)
    calculate_delta_ll(surprisal_ngram_5_alpha4, k, emotion_data_ngram_5_alpha4, ngram_std_5_alpha4)
    calculate_delta_ll(surprisal_ngram_2_alpha4, k, emotion_data_ngram_2_alpha20, ngram_std_2_alpha20)
    calculate_delta_ll(surprisal_ngram_3_alpha20, k, emotion_data_ngram_3_alpha20, ngram_std_3_alpha20)
    calculate_delta_ll(surprisal_ngram_4_alpha20, k, emotion_data_ngram_4_alpha20, ngram_std_4_alpha20)
    calculate_delta_ll(surprisal_ngram_5_alpha20, k, emotion_data_ngram_5_alpha20, ngram_std_5_alpha20)


def plot_data(emotion, emotion_data, std_data, plt_number, c):

  # Plot results for BERT model
  plt.subplot(2, 5, plt_number)

  y_axis = np.array(emotion_data[emotion])
  y_std = np.array(std_data[emotion]) 
  y_std = np.std(y_axis)

  # Adjust the size of the dots based on the standard deviation of y_axis
  dot_size = 100
  # Plot the scatter plot
  plt.scatter(x_axis, y_axis, s=dot_size, color=c)

  # Add shadows based on the standard deviation of y_axis
  shadow_c = c[:-1] + (0.3,)
  plt.fill_between(x_axis, y_axis + y_std, y_axis - y_std, color=shadow_c, label='Shadow')
  # Add a vertical line at the position of the maximum peak
  plt.title(emotion_names[emotion], fontsize=20)
  # Set x-axis ticks more frequently
  plt.xticks(np.linspace(0.25, 2.5, 4))  # Adjust the parameters as needed

  return


fig = plt.figure(figsize=(15,7))
emotion_names = ['неутрално', 'срећно', 'тужно', 'уплашено', 'љуто']

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_bert, bert_std, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_bertic, bertic_std, emotion + 1, (1, 0 , 0, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['BERT','BERT std', 'BERTic', 'BERTic std'])


fig = plt.figure(figsize=(15,7))
emotion_names = ['неутрално', 'срећно', 'тужно', 'уплашено', 'љуто']

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_gpt_2, gpt_std_2, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_gpt_3, gpt_std_3, emotion + 1, (1, 0 , 0, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['gpt-2','gpt-2 std', 'gpt-neo', 'gpt-neo std'])


fig = plt.figure(figsize=(15,7))

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_gpt_2, gpt_std_2, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_yugo, yugo_std, emotion + 1, (1, 0 , 0, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['gpt-2', 'gpt-2 std', 'yugo', 'yugo std'])



  
fig = plt.figure(figsize=(15,7))

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_ngram_2_alpha4, ngram_std_2_alpha4, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_ngram_3_alpha4, ngram_std_3_alpha4, emotion + 1, (0, 0 , 0, 1))
  plot_data(emotion, emotion_data_ngram_4_alpha4, ngram_std_4_alpha4, emotion + 1, (1, 0, 1, 1))
 # plot_data(emotion, emotion_data_ngram_5, ngram_std_5, emotion + 1, (0, 1, 0, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['2-gram','2-gram std', '3-gram', '3-gram std','4-gram', '4-gram std'])


  
fig = plt.figure(figsize=(15,7))

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_ngram_2_alpha20, ngram_std_2_alpha20, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_ngram_3_alpha20, ngram_std_3_alpha20, emotion + 1, (0, 0 , 0, 1))
  plot_data(emotion, emotion_data_ngram_4_alpha20, ngram_std_4_alpha20, emotion + 1, (1, 0, 1, 1))
 # plot_data(emotion, emotion_data_ngram_5, ngram_std_5, emotion + 1, (0, 1, 0, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['2-gram','2-gram std', '3-gram', '3-gram std','4-gram', '4-gram std'])


fig = plt.figure(figsize=(15,7))

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_ngram_4_alpha4, ngram_std_4_alpha4, emotion + 1, (1, 0, 1, 1))
  plot_data(emotion, emotion_data_gpt_2, gpt_std_2, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_yugo, yugo_std, emotion + 1, (1, 0 , 0, 1))
  plot_data(emotion, emotion_data_bert, bert_std, emotion + 1, (0, 0 , 0, 1))
  plot_data(emotion, emotion_data_bertic, bertic_std, emotion + 1, (0.5, 0, 0.5, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['4-gram','4-gram std', 'gpt-2','gpt-2 std', 'yugo', 'yugo std', 'bert', 'bert std', 'bertic', 'berti std'])



