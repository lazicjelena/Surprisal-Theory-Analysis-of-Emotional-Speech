# -*- coding: utf-8 -*-
"""my_functions.py

Created on Fri Nov 22 11:17:57 2024

@author: Jelena

Pipeline role
-------------
Library of regression and statistical-test helpers shared
across the ``Different information measurement parameters/``
folder. Provides the per-row sentence/word feature lookup
:func:`lookup_features`, the AIC and Gaussian log-likelihood
helpers (:func:`calculate_log_Likelihood`,
:func:`calculate_aic`, :func:`akaike_for_column`), the
"flexible" delta log-likelihood wrapper
:func:`calculate_delta_ll_flexible` (registered behind the
P-009 dispatcher :func:`calculate_delta_ll`), the paired
permutation test :func:`paired_permutation_test`, and three
regression front-ends used by the analyses:
:func:`add_column_with_surprisal`, :func:`fonetic_model` and
:func:`add_column`. All regressions are fold-wise on
``log2(time)`` with a 3-sigma outlier filter on the training
fold; predictions are written into per-row prediction columns
and the calling analyses then compare them to the
``baseline -k`` columns produced by :func:`add_column`.
"""
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import numpy as np
import math 
import pandas as pd
from utils.stats_utils import calculate_log_Likelihood, calculate_aic

def lookup_features(data, freq_df, column_name):
    """Build a per-row summed surprisal value keyed by ``(target sentence, word)``.

    For every row of ``data`` the words in ``data['word']``
    (a single spoken token, possibly hyphenated as multiple
    orthographic words separated by spaces) are looked up in
    ``freq_df`` filtered by the matching ``Sentence``. Repeated
    occurrences of the same word inside a sentence are resolved
    positionally by counting how many times the same word has
    already been consumed in the current sentence; if the
    positional lookup raises (e.g. fewer rows than expected),
    the function falls back to the first matching row. Missing
    words contribute ``0`` and trigger an ``error`` print.

    Parameters
    ----------
    data : pandas.DataFrame
        Master per-word table with at least ``word`` and
        ``target sentence`` columns.
    freq_df : pandas.DataFrame
        Per-(sentence, word) lookup table with at least
        ``Sentence``, ``Word`` and ``column_name`` columns.
    column_name : str
        Name of the surprisal/log-probability column to extract.

    Returns
    -------
    list of float
        One summed value per row of ``data``, in input order.
    """
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

# Calculate AIC for models with different numbers of parameters

def akaike_for_column(data, model_name, baseline_model = 'baseline'):
    """Return ``mean_ll(baseline) - mean_ll(model)`` plus the model's std.

    Both predictions and observations are kept on the linear
    ``time`` scale; the baseline column is assumed to have been
    written by :func:`add_column` (e.g. ``"baseline -3"``) and
    ``model_name`` is the surprisal/parameter-augmented column
    written by :func:`add_column_with_surprisal` or
    :func:`fonetic_model`. Rows where either prediction is
    missing are dropped before scoring. Penalty parameter
    counts are hard-coded to ``2`` for the baseline and ``3``
    for the augmented model.

    Parameters
    ----------
    data : pandas.DataFrame
        Master table containing ``time``, ``baseline_model`` and
        ``model_name`` columns.
    model_name : str
        Name of the augmented prediction column.
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
    
    data = data.dropna(subset=[model_name, baseline_model])
    _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2

    return difference, std_ll_2

def calculate_delta_ll_flexible(data, model_name, baseline = "baseline -3"):
    """Best-effort wrapper around :func:`akaike_for_column`.

    Catches all exceptions from
    :func:`akaike_for_column(data, model_name, baseline)` and
    returns ``(0, 0)`` instead, after printing an error
    message; this keeps the per-``(speaker, emotion, parameter)``
    sweep going when a slice is empty or otherwise
    ill-conditioned.

    Parameters
    ----------
    data : pandas.DataFrame
        Slice of the master table for a single
        ``(speaker, emotion)`` (or similar) pair.
    model_name : str
        Name of the augmented prediction column.
    baseline : str, optional
        Name of the baseline prediction column. Defaults to
        ``"baseline -3"``.

    Returns
    -------
    delta_ll : float
        Mean log-likelihood improvement over baseline; ``0`` on
        failure.
    std_element : float
        Standard deviation of the model's per-point
        log-likelihood; ``0`` on failure.
    """

    try:
      delta_ll, std_element = akaike_for_column(data, model_name, baseline)
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {model_name}")
      return 0, 0


def calculate_delta_ll(mode, **kwargs):
    """Dispatcher za calculate_delta_ll varijante u ovom fajlu (P-009).

    Dostupni mode-ovi u Different information measurement parameters/my_functions.py:
      - "flexible"  → calculate_delta_ll_flexible(data, model_name, baseline="baseline -3")
    """
    mapping = {
        "flexible": calculate_delta_ll_flexible,
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return mapping[mode](**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for mode '{mode}': {e}")


def paired_permutation_test(df, col1, col2, num_permutations=1000):
    """Paired permutation test on the mean difference of two columns.

    Builds the observed mean difference ``mean(scores1 - scores2)``
    over the paired rows of ``df[col1]`` and ``df[col2]``, then
    runs ``num_permutations`` random sign-flips on the per-pair
    differences and counts the fraction of permuted absolute
    mean differences that exceed the observed value. The
    returned p-value is therefore two-sided.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing the two prediction columns.
    col1, col2 : str
        Column names of the two prediction series.
    num_permutations : int, optional
        Number of random sign-flip permutations. Defaults to
        ``1000``.

    Returns
    -------
    float
        Two-sided permutation p-value.
    """

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
    """Fold-wise linear regression with phonetic features added on top of GPT-2 surprisal.

    Uses the predictors ``[length, log probability,
    Surprisal GPT-2]`` plus their per-row lags ``-1, -2, -3``
    plus ``fonem_list`` and trains one
    :class:`sklearn.linear_model.LinearRegression` per fold on
    ``log2(time)`` (3-sigma outlier filter on the training
    fold). Out-of-fold predictions are written to a
    ``"fonetic model"`` column in the returned DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Master table with at least ``length``, ``log probability``,
        ``Surprisal GPT-2``, ``time``, ``fold`` and the columns
        listed in ``fonem_list``.
    fonem_list : list of str
        Phonetic feature column names to include as predictors.

    Returns
    -------
    pandas.DataFrame
        Concatenation of the per-fold test slices with the
        ``"fonetic model"`` prediction column added (duplicates
        dropped).
    """
    
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
    """Surprisal-free fold-wise baseline regression.

    Fits a fold-wise :class:`sklearn.linear_model.LinearRegression`
    on ``log2(time)`` with predictors
    ``[length, log probability]`` (plus their per-row lags
    ``-1..-k`` when ``k > 0``) and a 3-sigma outlier filter on
    the training fold. Out-of-fold predictions are written to a
    ``"baseline -k"`` column in the returned DataFrame and the
    per-fold mean squared error on the held-out test slice is
    averaged and printed for diagnostics.

    Parameters
    ----------
    df : pandas.DataFrame
        Master table with ``length``, ``log probability``,
        ``time`` and ``fold`` columns.
    k : int, optional
        Lag depth (default ``0``). When non-zero, predictors
        ``length -i`` and ``log probability -i`` for
        ``i = 1..k`` are included.

    Returns
    -------
    pandas.DataFrame
        Concatenation of the per-fold test slices with the
        ``"baseline -k"`` prediction column added (duplicates
        dropped).
    """
    
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
