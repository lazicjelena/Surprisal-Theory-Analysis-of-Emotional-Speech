# -*- coding: utf-8 -*-
"""surprisal_results.py

Created on Sun Oct 20 19:37:25 2024

@author: Jelena

Pipeline role
-------------
Main duration-regression analysis script for the
``Duration Prediction based on Surprisals/`` folder. For each of
the five surprisal models (GPT-2, Yugo, ngram-3, BERT, BERTic),
loads ``../podaci/training data/<Surprisal X>.csv`` (built by
``build_surprisal_datasets.py``), merges in the surprisal-free
baseline predictions from
``../podaci/results - linear regression/baseline_results_data.csv``
(produced by ``baseline_model.py``), and sweeps the surprisal
exponent ``k`` in ``np.arange(0.25, 3, 0.25)``. For every ``k``
the helper :func:`inf_k_model` adds an
``<Surprisal X> <k>`` raised-power column and trains a fold-wise
:class:`LinearRegression` on
``[length, log probability, <Surprisal X> <k>]``. Per
``(speaker gender, emotion, k)`` the dispatcher
:func:`calculate_delta_ll` (mode ``"global"``) computes the
log-likelihood improvement over baseline (Gaussian
log-likelihood derived from residuals via :func:`calculate_aic`).
The summarised tables are written to
``../podaci/results - linear regression/<Surprisal X>_results.csv``
and consumed by ``final_graphs.py``.
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import numpy as np
import pandas as pd
import warnings
import os
import math 
from utils.stats_utils import calculate_log_Likelihood, calculate_aic


def inf_k_model(df, k, surprisal):
    """Add a ``surprisal**k`` column and store fold-wise predictions.

    For the given surprisal column ``surprisal`` and exponent
    ``k`` an auxiliary column ``"<surprisal> <k>"`` is added with
    the per-row value ``df[surprisal] ** k``, then a per-fold
    :class:`LinearRegression` is fit on ``log2(time)`` with
    predictors ``[length, log probability, "<surprisal> <k>"]``
    (3-sigma outlier filter on the training fold). Out-of-fold
    predictions are written to a ``"<surprisal> <k> model"``
    column in the returned DataFrame; the auxiliary
    ``"<surprisal> <k>"`` column is dropped before
    concatenation.

    Parameters
    ----------
    df : pandas.DataFrame
        Master table with at least ``length``, ``log probability``,
        ``time``, ``fold`` and ``surprisal`` columns.
    k : float
        Surprisal exponent.
    surprisal : str
        Name of the surprisal column to raise.

    Returns
    -------
    pandas.DataFrame
        Concatenation of the per-fold test slices with the
        ``"<surprisal> <k> model"`` prediction column added.
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

# Calculate AIC for models with different numbers of parameters

def akaike_for_column(data, model_name, baseline_model = 'baseline'):
    """Return ``mean_ll(baseline) - mean_ll(model)`` plus the model's std.

    Both predictions and observations are kept on the linear
    ``time`` scale (no ``log2`` is applied here); the baseline
    column is assumed to have been written by
    ``baseline_model.py`` and ``model_name`` is the
    ``"<Surprisal X> <k> model"`` column produced by
    :func:`inf_k_model`. Penalty parameter counts are hard-coded
    to ``2`` for the baseline (``length`` + ``log probability``)
    and ``3`` for the surprisal-augmented model.

    Parameters
    ----------
    data : pandas.DataFrame
        Master table containing ``time``, ``baseline_model`` and
        ``model_name`` columns.
    model_name : str
        Name of the surprisal-augmented prediction column.
    baseline_model : str, optional
        Name of the baseline prediction column. Defaults to
        ``"baseline"``.

    Returns
    -------
    difference : float
        ``mean_ll(baseline) - mean_ll(model)``.
    std_ll_2 : float
        Standard deviation of the model's per-point
        log-likelihood.
    """
    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2

    return difference, std_ll_2

def calculate_delta_ll_global(data, surprisal_name, k):
    """Best-effort wrapper around :func:`akaike_for_column`.

    Catches all exceptions from
    :func:`akaike_for_column(data, "<surprisal> <k> model", "baseline")`
    and returns ``(0, 0)`` instead, after printing an error
    message; this keeps the per-``(gender, emotion, k)`` sweep
    going when an emotion / gender slice is empty or otherwise
    ill-conditioned.

    Parameters
    ----------
    data : pandas.DataFrame
        Slice of the master table for a single
        ``(gender, emotion)`` pair.
    surprisal_name : str
        Base name of the surprisal column (e.g.
        ``"Surprisal GPT-2"``).
    k : float
        Surprisal exponent.

    Returns
    -------
    delta_ll : float
        Mean log-likelihood improvement of the surprisal model
        over the baseline; ``0`` on failure.
    std_element : float
        Standard deviation of the surprisal model's per-point
        log-likelihood; ``0`` on failure.
    """
    try:
      delta_ll, std_element = akaike_for_column(data, surprisal_name + ' ' + str(k) + ' model', 'baseline')
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal} at k = {k}")
      return 0, 0


def calculate_delta_ll(mode, **kwargs):
    """Dispatcher za calculate_delta_ll varijante u ovom fajlu (P-009).

    Dostupni mode-ovi u Duration Prediction based on Surprisals/surprisal_results.py:
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
                delta_element, std_element = calculate_delta_ll(mode="global", data=emotion_data, surprisal_name=surprisal, k=k)
                
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





