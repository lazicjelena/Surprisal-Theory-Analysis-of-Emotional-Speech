# -*- coding: utf-8 -*-
"""surprisal_results.py

Created on Sun Oct 20 19:37:25 2024

@author: Jelena

Pipeline role
-------------
Per-surprisal lag-aware ``\Delta\log\mathcal{L}`` aggregator for
the split-over analysis. For each surprisal channel in
``['Surprisal GPT-2', 'Surprisal Yugo', 'Surprisal ngram-3',
'Surprisal BERT', 'Surprisal BERTic']`` and for ``k`` in
``[0..4]``, reads the lag-augmented features from
``../podaci/split-over data/<surprisal>.csv`` (built by
``build_surprisal_datasets.py``) and joins them with the
``baseline -k`` columns from
``../podaci/split-over results/baseline_results_data.csv`` (built
by ``Split-over effect/baseline_model.py``). Fits a per-fold
linear regression of ``log2(time)`` on
``[length, log probability, surprisal]`` plus all lag columns up
to ``k`` (:func:`add_column_with_surprisal`), then computes the
per-emotion / per-gender ``\Delta\log\mathcal{L}`` against
``baseline -k`` via :func:`calculate_delta_ll_lag` (dispatched
through the P-009 :func:`calculate_delta_ll`). The
``(y_axis, y_std, k, emotion, speaker gender)`` summary is saved
to ``../podaci/split-over results/<surprisal>_results.csv``,
which is the input to ``final_graphs.py``.
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import numpy as np
import pandas as pd
import os
import math 

def calculate_log_Likelihood(data):
    """Per-element normal log-pdf using the empirical mean / std of ``data``.

    Local copy of the helper from
    ``Linear regression/stats_utils.py``. Estimates
    ``mu = mean(data)``, ``sigma = std(data)`` and returns
    ``norm.logpdf(data, loc=mu, scale=sigma)``.

    Parameters
    ----------
    data : array_like of float
        Input vector (typically a residual array).

    Returns
    -------
    numpy.ndarray
        Per-element log-density values; ``shape == data.shape``.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.logpdf(data, loc=mean, scale=std_dev)

# Calculate AIC for models with different numbers of parameters
def calculate_aic(real_values, results, k):
    """AIC plus mean / std log-likelihood for one model column.

    Local copy of the helper from
    ``Linear regression/stats_utils.py``. Computes the residual
    array, scores it under a Gaussian noise model with
    :func:`calculate_log_Likelihood`, and forms the per-row AIC.

    Parameters
    ----------
    real_values : array_like of float
        Observed durations.
    results : array_like of float
        Model predictions, same shape as ``real_values``.
    k : int
        Number of effective parameters of the model.

    Returns
    -------
    tuple of (numpy.ndarray, float, float)
        ``(aic_per_row, mean_log_likelihood, std_log_likelihood)``.
    """
    residuals = np.array(real_values) - np.array(results)
    log_likelihood = calculate_log_Likelihood(residuals)
    aic = 2 * k - 2 * log_likelihood
    return aic, np.mean(log_likelihood), np.std(log_likelihood)

def akaike_for_column(data, model_name, baseline_model = 'baseline'):
    """Compute the per-column ``\Delta\log\mathcal{L}`` baseline vs. model.

    Drops rows with NaNs in either the model or baseline column
    (lag features can introduce edge-of-sentence NaNs), then calls
    :func:`calculate_aic` once on each (with ``k=2`` for baseline,
    ``k=3`` for model). Returns the ``baseline - model`` mean
    log-likelihood difference and the std of the model
    log-likelihoods. Note: ``baseline_model`` defaults to plain
    ``'baseline'``; the call sites in this script pass the
    lag-specific ``f"baseline -{k}"`` column.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-emotion slice with ``time``, the lag-baseline column
        and the model column.
    model_name : str
        Column name of the surprisal-augmented model predictions.
    baseline_model : str, optional
        Column name of the surprisal-free baseline predictions.
        Defaults to ``'baseline'``; in this script the caller
        supplies a lag-specific column instead.

    Returns
    -------
    tuple of (float, float)
        ``(mean_ll_baseline - mean_ll_model, std_ll_model)``.
    """
    data = data.dropna(subset=[model_name, baseline_model])
    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2

    return difference, std_ll_2

def calculate_delta_ll_lag(data, surprisal_name, k):
    """Per-emotion lag-``k`` ``\Delta\log\mathcal{L}`` for one ``(surprisal, k)``.

    Wraps :func:`akaike_for_column` with the lag-specific
    ``f"{surprisal_name} -{k} model"`` model column and
    ``f"baseline -{k}"`` reference column. On any internal
    exception (typically a missing model column for a degenerate
    fold), prints a warning and returns ``(0, 0)`` so the caller's
    sweep loop can continue.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-emotion / per-gender slice with ``time``, the
        lag-baseline column and the surprisal model column.
    surprisal_name : str
        Surprisal label (e.g. ``'Surprisal GPT-2'``).
    k : int
        Lag order; both the model and baseline column names use
        ``-{k}``.

    Returns
    -------
    tuple of (float, float)
        ``(delta_ll, std_ll_model)``, or ``(0, 0)`` on failure.
    """
    try:
      delta_ll, std_element = akaike_for_column(data, f"{surprisal_name} -{k} model", f"baseline -{k}")
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal} at k = {k}")
      return 0, 0


def calculate_delta_ll(mode, **kwargs):
    """Dispatcher za calculate_delta_ll varijante u ovom fajlu (P-009).

    Dostupni mode-ovi u Split-over effect/surprisal_results.py:
      - "lag"  → calculate_delta_ll_lag(data, surprisal_name, k)
    """
    mapping = {
        "lag": calculate_delta_ll_lag,
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return mapping[mode](**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for mode '{mode}': {e}")
    

def add_column_with_surprisal(df, surprisal, k=0):
    """Per-fold ``log2(time)`` regression with surprisal + lag predictors.

    Builds the predictor list
    ``['length', 'log probability', surprisal] +
    ['length -i', 'log probability -i', f"{surprisal} -i"]``
    for ``i in 1..k``, drops rows with any NaN in the predictors,
    then fits leave-one-fold-out linear regressions of
    ``log2(time)`` on those predictors. Training side is filtered
    with a 3-sigma rule on ``log2(time)``. Per-row out-of-fold
    predictions are written into a new column
    ``f"{surprisal} -{k} model"``. The function then returns; the
    duplicated trailing block in the source is unreachable.

    Parameters
    ----------
    df : pandas.DataFrame
        Master split-over table for one surprisal channel,
        already merged with the baseline-results table.
    surprisal : str
        Name of the surprisal column.
    k : int, optional
        Maximum lag to include. ``0`` means no lag columns.

    Returns
    -------
    pandas.DataFrame
        Concatenation of all out-of-fold predictions, with the
        added ``f"{surprisal} -{k} model"`` column and duplicates
        dropped.
    """
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
            
                delta_element, std_element = calculate_delta_ll(mode="lag", data=emotion_data, surprisal_name=surprisal, k=k)
                
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





