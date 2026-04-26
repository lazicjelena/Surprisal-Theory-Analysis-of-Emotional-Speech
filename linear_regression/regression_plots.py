# -*- coding: utf-8 -*-
"""final_graphs.py

Created on Fri Aug 23 05:22:39 2024

@author: Jelena

Pipeline role
-------------
Final ``\Delta\log\mathcal{L}`` plotting script for the global
duration regression. Reads the master training table
``../podaci/training_data.csv`` (built by ``build_dataset.py`` and
augmented with ``baseline`` by ``baseline_model.py``), and for a
sweep of surprisal-power exponents ``k`` in ``[0.25, 0.5, ..., 2.75]``
fits per-fold linear regressions of ``log2(time)`` on
``length``, ``log probability`` and ``surprisal**k``
(:func:`inf_k_model`), then computes per-emotion / per-gender
``\Delta\log\mathcal{L}`` against ``baseline`` via
:func:`akaike_for_column` / :func:`calculate_delta_ll_global`
(dispatched through the P-009 :func:`calculate_delta_ll`). Renders
the four publication figures (English / Serbian, autoregressive /
bidirectional model groups). Output is purely visual; no CSVs are
written from this script.
"""

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import math
import warnings

from utils.stats_utils import calculate_log_Likelihood, calculate_aic

def inf_k_model(df, k, surprisal):
    """Fit ``log2(time) ~ length + log_prob + surprisal**k`` per fold.

    Adds a new column ``f"{surprisal} {k}"`` to ``df`` holding
    ``df[surprisal] ** k`` (the power-transformed surprisal) and
    runs a leave-one-fold-out linear regression of ``log2(time)``
    on the three predictors ``length``, ``log probability``, and
    that new column. Per-fold predictions are stitched into
    ``f"{surprisal} {k} model"`` and the power-transformed column
    is dropped before the row is concatenated, so the returned
    DataFrame has the original columns plus the model-prediction
    column. Outliers are filtered on the training side with a
    3-sigma rule on ``log2(time)``.

    Parameters
    ----------
    df : pandas.DataFrame
        Master training table; must contain ``length``,
        ``log probability``, ``time``, ``fold``, and the named
        ``surprisal`` column.
    k : float
        Surprisal-power exponent. The new feature column is named
        ``f"{surprisal} {k}"``.
    surprisal : str
        Surprisal column name to power-transform (e.g.
        ``'surprisal GPT'``).

    Returns
    -------
    pandas.DataFrame
        Concatenation of all out-of-fold predictions; one row per
        original row of ``df``, with new column
        ``f"{surprisal} {k} model"``.
    """
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

def akaike_for_column(data, model_name, baseline_model = 'baseline'):
    """Compute the per-column ``\Delta\log\mathcal{L}`` baseline vs. model.

    Calls :func:`calculate_aic` twice on the same ``data['time']``
    column: once against ``data[baseline_model]`` (with ``k=2``)
    and once against ``data[model_name]`` (with ``k=3``). Returns
    the difference of mean log-likelihoods and the std of the
    model log-likelihoods. The convention is ``baseline - model``,
    so a positive ``difference`` means the surprisal-augmented
    model fits worse than baseline (and a negative value means it
    helps).

    Parameters
    ----------
    data : pandas.DataFrame
        Holds ``time`` and the two prediction columns.
    model_name : str
        Column name of the surprisal-augmented model predictions.
    baseline_model : str, optional
        Column name of the surprisal-free baseline predictions.
        Defaults to ``'baseline'``.

    Returns
    -------
    tuple of (float, float)
        ``(mean_ll_baseline - mean_ll_model, std_ll_model)``.
    """
    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2

    return difference, std_ll_2

def calculate_delta_ll_global(data, surprisal_name, k):
    """Global per-emotion ``\Delta\log\mathcal{L}`` for one ``(surprisal, k)``.

    Wraps :func:`akaike_for_column` with the conventional
    ``f"{surprisal_name} {k} model"`` model column and the
    canonical ``'baseline'`` reference column. On any internal
    exception (typically a missing model column for a degenerate
    fold), prints a warning and returns ``(0, 0)`` so the caller's
    plotting loop can continue.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-emotion / per-gender slice with ``time``, ``baseline``,
        and the model column.
    surprisal_name : str
        Surprisal label (e.g. ``'surprisal GPT'``).
    k : float
        Surprisal-power exponent used to build the model column.

    Returns
    -------
    tuple of (float, float)
        ``(delta_ll, std_ll_model)`` from
        :func:`akaike_for_column`, or ``(0, 0)`` on failure.
    """
    try:
      delta_ll, std_element = akaike_for_column(data, surprisal_name + ' ' + str(k) + ' model', 'baseline')
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal} at k = {k}")
      return 0, 0


def calculate_delta_ll(mode, **kwargs):
    """Dispatcher za calculate_delta_ll varijante u ovom fajlu (P-009).

    Dostupni mode-ovi u Linear regression/final_graphs.py:
      - "global"  → calculate_delta_ll_global(data, surprisal_name, k)
    """
    mapping = {
        "global": calculate_delta_ll_global,
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return mapping[mode](**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for mode '{mode}': {e}")
    
    
    
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
                delta_element,  std_element = calculate_delta_ll(mode="global", data=emotion_data, surprisal_name=surprisal, k=k)
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
                delta_element,  std_element = calculate_delta_ll(mode="global", data=emotion_data, surprisal_name=surprisal, k=k)
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
                delta_element,  std_element = calculate_delta_ll(mode="global", data=emotion_data, surprisal_name=surprisal, k=k)
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
                delta_element,  std_element = calculate_delta_ll(mode="global", data=emotion_data, surprisal_name=surprisal, k=k)
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