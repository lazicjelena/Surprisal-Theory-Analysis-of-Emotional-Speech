# -*- coding: utf-8 -*-
"""my_functions.py

Created on Wed Feb 19 22:25:34 2025

@author: Jelena

Pipeline role
-------------
Library of regression and statistical-test helpers shared
across the ``Additional files after recension/`` folder. The
core regression entry point is :func:`inf_k_model`, which
trains a fold-wise linear model on a configurable surprisal
function form (power, linear, logarithmic, exponential) on
top of ``[length, log probability]``. The Gaussian
log-likelihood / AIC helpers
(:func:`calculate_log_Likelihood`, :func:`calculate_aic`,
:func:`akaike_for_column`) and the per-emotion /
per-prominence delta log-likelihood wrappers
(:func:`calculate_delta_ll_emotion_prominence`,
:func:`calculate_delta_ll_prominence`, registered behind the
P-009 dispatcher :func:`calculate_delta_ll`) consume those
predictions to score regressions against a baseline. The
feature-joiner :func:`lookup_features`, the grammatical-type
joiner :func:`add_word_type` and the canonical-sentence
matcher :func:`most_similar_sentence_index` are used by the
dataset builders upstream of the regression analyses.
"""

from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import numpy as np
import pandas as pd
import math 

def inf_k_model(df, k, surprisal, prosody = 'time', function = 'power'):
    """Add a transformed surprisal column and store fold-wise predictions.

    For the given surprisal column ``surprisal`` and exponent
    ``k`` an auxiliary column ``"<surprisal> <k>"`` is added
    with the per-row value transformed by ``function``
    (``"power"`` -> ``surprisal ** k``, ``"linear"`` ->
    ``surprisal * k``, ``"logarithmic"`` ->
    ``np.log(surprisal)``, ``"exponential"`` ->
    ``np.exp(surprisal)``). A per-fold
    :class:`sklearn.linear_model.LinearRegression` is then fit
    on ``log2(prosody)`` with predictors ``[length,
    log probability, "<surprisal> <k>"]`` (3-sigma outlier
    filter on the training fold). Out-of-fold predictions are
    written to a ``"<surprisal> <k> model"`` column (with the
    function name appended when ``function != "power"``).

    Parameters
    ----------
    df : pandas.DataFrame
        Master table with at least ``length``, ``log probability``,
        ``prosody``, ``fold`` and ``surprisal`` columns.
    k : float
        Surprisal exponent / coefficient.
    surprisal : str
        Name of the surprisal column to transform.
    prosody : str, optional
        Name of the regression target column. Defaults to
        ``"time"``.
    function : str, optional
        Surprisal transform: ``"power"`` (default), ``"linear"``,
        ``"logarithmic"`` or ``"exponential"``.

    Returns
    -------
    pandas.DataFrame
        Concatenation of the per-fold test slices with the
        ``"<surprisal> <k> model"`` (plus ``function`` suffix
        when non-power) prediction column added.
    """

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
    """Pointwise Gaussian log-likelihood of ``data`` under its own MLE.

    The maximum-likelihood Gaussian fit (mean and standard
    deviation taken from ``data``) is evaluated at every input
    point; no aggregation is performed.

    Parameters
    ----------
    data : array-like
        One-dimensional sample of residuals.

    Returns
    -------
    numpy.ndarray
        Per-element log-density values.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return norm.logpdf(data, loc=mean, scale=std_dev)

# Calculate AIC for models with different numbers of parameters
def calculate_aic(real_values, results, k):
    """Compute the Akaike Information Criterion plus log-likelihood summaries.

    The residuals ``real_values - results`` are scored via
    :func:`calculate_log_Likelihood`; the per-point AIC is
    returned together with the mean and standard deviation of
    the log-likelihood across all samples. Only the mean and
    standard deviation are used downstream by
    :func:`akaike_for_column`.

    Parameters
    ----------
    real_values : array-like
        Observed values (typically ``log2(time)``).
    results : array-like
        Model predictions, aligned with ``real_values``.
    k : int
        Number of free parameters in the model (used for the
        ``2 * k`` AIC penalty term).

    Returns
    -------
    aic : numpy.ndarray
        Per-point AIC values.
    mean_log_likelihood : float
        Mean of the per-point log-likelihoods.
    std_log_likelihood : float
        Standard deviation of the per-point log-likelihoods.
    """
    
    residuals = np.array(real_values) - np.array(results)
    log_likelihood = calculate_log_Likelihood(residuals)
    aic = 2 * k - 2 * log_likelihood
    return aic, np.mean(log_likelihood), np.std(log_likelihood)

def akaike_for_column(data, prominence, model_name, baseline_model = 'baseline'):
    """Return ``mean_ll(baseline) - mean_ll(model)`` plus the std difference.

    Both predictions and observations are kept on the linear
    ``prominence`` scale (no ``log2`` applied here); the
    baseline column is assumed to have been written by
    ``baseline_model.py`` / ``baseline_model_prosody.py`` and
    ``model_name`` is the surprisal-augmented column produced
    by :func:`inf_k_model`. Penalty parameter counts are
    hard-coded to ``2`` for the baseline (``length`` +
    ``log probability``) and ``3`` for the surprisal-augmented
    model.

    Parameters
    ----------
    data : pandas.DataFrame
        Master table containing ``prominence``,
        ``baseline_model`` and ``model_name`` columns.
    prominence : str
        Name of the regression target column (e.g. ``"time"``,
        ``"energy"``, ``"f0"``).
    model_name : str
        Name of the surprisal-augmented prediction column.
    baseline_model : str, optional
        Name of the baseline prediction column. Defaults to
        ``"baseline"``.

    Returns
    -------
    difference : float
        ``mean_ll(baseline) - mean_ll(model)``.
    std_difference : float
        ``std_ll(baseline) - std_ll(model)``.
    """

    _, mean_ll_1, std_ll_1 = calculate_aic(data[prominence], data[baseline_model], 2)
    _, mean_ll_2, std_ll_2 = calculate_aic(data[prominence], data[model_name], 3)
    difference = mean_ll_1 - mean_ll_2
    std_difference = std_ll_1 - std_ll_2

    return difference, std_difference


def calculate_delta_ll_emotion_prominence(data, surprisal, k, emotion_data, std_data, prominence = 'time', function = 'power'):
    """Per-emotion delta log-likelihood appender for the prominence sweep.

    Wraps :func:`akaike_for_column` for the
    ``"<surprisal> <k> model"`` (plus ``function`` suffix when
    non-power) column; the returned vector of per-emotion
    delta log-likelihoods and the matching std vector are
    appended in-place to ``emotion_data`` and ``std_data``
    (one list per emotion). On any exception the routine
    silently appends sentinel zeros / ones so the calling
    sweep can keep going on ill-conditioned slices.

    Parameters
    ----------
    data : pandas.DataFrame
        Slice of the master table for the current
        ``(model, k, function)`` cell.
    surprisal : str
        Base name of the surprisal column.
    k : float
        Surprisal exponent / coefficient.
    emotion_data : list of list
        Per-emotion accumulator of delta log-likelihoods
        (mutated in place).
    std_data : list of list
        Per-emotion accumulator of std values (mutated in place).
    prominence : str, optional
        Regression target column. Defaults to ``"time"``.
    function : str, optional
        Surprisal transform name (used to suffix the model
        column). Defaults to ``"power"``.

    Returns
    -------
    None
    """


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

def calculate_delta_ll_prominence(data, surprisal_name, k, prominence = 'time', function = 'power'):
    """Best-effort wrapper around :func:`akaike_for_column` for the prominence sweep.

    Catches all exceptions from
    :func:`akaike_for_column(data, prominence, model_name,
    "baseline")` and returns ``(0, 0)`` instead, after printing
    an error message; this keeps the per-``(surprisal, k,
    function)`` sweep going when a slice is empty or otherwise
    ill-conditioned.

    Parameters
    ----------
    data : pandas.DataFrame
        Slice of the master table for the current sweep cell.
    surprisal_name : str
        Base name of the surprisal column.
    k : float
        Surprisal exponent / coefficient.
    prominence : str, optional
        Regression target column. Defaults to ``"time"``.
    function : str, optional
        Surprisal transform name. Defaults to ``"power"``.

    Returns
    -------
    delta_ll : float
        Mean log-likelihood improvement over baseline; ``0``
        on failure.
    std_element : float
        Std difference; ``0`` on failure.
    """

    model_name = surprisal_name + ' ' + str(k) + ' model'
    if function != 'power':
        model_name+= function

    try:
      delta_ll, std_element = akaike_for_column(data, prominence, model_name, 'baseline')
      return delta_ll, std_element
    except:
      print(f"Error accured while processing {surprisal_name} at k = {k}")
      return 0, 0


def calculate_delta_ll(mode, **kwargs):
    """Dispatcher za calculate_delta_ll varijante u ovom fajlu (P-009).

    Dostupni mode-ovi u Additional files after recension/my_functions.py:
      - "emotion_prominence"  → calculate_delta_ll_emotion_prominence(data, surprisal, k, emotion_data, std_data, prominence='time', function='power')
      - "prominence"          → calculate_delta_ll_prominence(data, surprisal_name, k, prominence='time', function='power')
    """
    mapping = {
        "emotion_prominence": calculate_delta_ll_emotion_prominence,
        "prominence": calculate_delta_ll_prominence,
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return mapping[mode](**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for mode '{mode}': {e}")


def lookup_features(data, freq_df, column_name):
    """Build a per-row summed surprisal value keyed by ``(target sentence, word)``.

    For every row of ``data`` the words in ``data['word']``
    (a single spoken token, possibly hyphenated as multiple
    orthographic words separated by spaces) are looked up in
    ``freq_df`` filtered by the matching ``Sentence``. When
    exactly one row matches, that row is taken; when multiple
    rows match (repeated words inside one sentence), the
    ``i``-th occurrence of the word picks the ``i``-th
    matching row. Missing words break out of the inner loop;
    the running sum at that point is recorded for the row.

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
    """Build per-row word-type strings keyed by ``(target sentence, word)``.

    For every row of ``data`` the words in ``data['word']``
    are looked up in ``freq_df`` filtered by the matching
    ``Sentence``. Repeated occurrences of the same word inside
    a sentence are resolved positionally (the ``i``-th
    occurrence picks the ``i``-th matching row). The returned
    list contains a space-joined concatenation of the
    ``column_name`` values across all space-separated
    sub-words of a row.

    Parameters
    ----------
    data : pandas.DataFrame
        Master per-word table with at least ``word`` and
        ``target sentence`` columns.
    freq_df : pandas.DataFrame
        Per-(sentence, word) lookup table with at least
        ``Sentence``, ``Word`` and ``column_name`` columns.
    column_name : str
        Name of the categorical column in ``freq_df`` to
        extract (typically ``"Type"``).

    Returns
    -------
    list of str
        One space-joined word-type string per row of ``data``,
        in input order.
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
    """Return the index of the canonical sentence with the most overlapping characters.

    The input ``sentence`` is lower-cased and stripped of
    spaces, then for every row of ``target_sentence_df`` the
    intersection size of the character sets is computed (via a
    nested ``common_chars`` helper). The row with the maximum
    intersection wins; the column ``"Similarity"`` is added in
    place to ``target_sentence_df`` as a side effect.

    Parameters
    ----------
    sentence : str
        Reconstructed utterance text.
    target_sentence_df : pandas.DataFrame
        Canonical target-sentence inventory with a ``Text``
        column (mutated in place to add ``"Similarity"``).

    Returns
    -------
    int
        Index of the most similar canonical sentence in
        ``target_sentence_df``.
    """
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