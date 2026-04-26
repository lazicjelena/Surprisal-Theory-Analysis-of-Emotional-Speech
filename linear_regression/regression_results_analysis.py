# -*- coding: utf-8 -*-
"""results.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta samo plotuje sve rezultate onako kako su prikazani u radu.

Pipeline role
-------------
Per-emotion ``\Delta\log\mathcal{L}`` plotting script for the
duration regression. Reads the master training table
``../podaci/training_data.csv`` (built by ``build_dataset.py`` and
augmented with ``baseline`` by ``baseline_model.py``), restricts to
male speakers and to non-``function``/non-``content`` word types,
and -- for a sweep of ``k`` values in ``[0.25 .. 2.75]`` --
power-transforms each surprisal channel via :func:`inf_k_model`.
Per-(emotion, k) ``\Delta\log\mathcal{L}`` is then accumulated
through :func:`calculate_delta_ll_emotion`/:func:`akaike_for_column`
(dispatched through the P-009 :func:`calculate_delta_ll`) into the
``emotion_data_*`` / ``*_std`` dictionaries, which are finally
rendered as the publication figures (BERT vs. BERTic, GPT-2 vs.
GPT-Neo, GPT-2 vs. Yugo, n-gram alpha=4 / alpha=20, ...). Output
is purely visual; no CSVs are written.
"""
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import math

from stats_utils import calculate_log_Likelihood, calculate_aic

def inf_k_model(df, k, surprisal):
    """Fit ``log2(time) ~ length + log_prob + surprisal**k`` per fold.

    Twin of :func:`residual_distribution.inf_k_model`. Adds
    ``f"{surprisal} {k}"`` (the power-transformed surprisal) and
    ``f"{surprisal} {k} model"`` (the fold-out predictions) to
    ``df``, runs leave-one-fold-out linear regression on
    ``log2(time)`` over ``length``, ``log probability`` and the
    transformed surprisal, and returns the concatenation of all
    out-of-fold rows. The training side is filtered with a 3-sigma
    rule on ``log2(time)``.

    Parameters
    ----------
    df : pandas.DataFrame
        Master training table.
    k : float
        Surprisal-power exponent.
    surprisal : str
        Surprisal column name to power-transform.

    Returns
    -------
    pandas.DataFrame
        Concatenation of all out-of-fold predictions; one row per
        original row of ``df`` plus the two new columns.
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
        results_df = pd.concat([results_df, test_data], axis=0)
        
    return results_df

def akaike_for_column(column_name, model_name, baseline_model = 'baseline'):
    """Group-wise ``\Delta\log\mathcal{L}`` over a categorical column.

    For every unique value of ``df[column_name]`` (in this script
    the categorical predictor of interest is ``'emotion'``),
    selects the matching subset of the *module-level* ``df`` and
    calls :func:`calculate_aic` once with ``baseline_model``
    (``k=2``) and once with ``model_name`` (``k=3``). The
    ``baseline - model`` mean log-likelihood difference is
    accumulated into a per-group list. Reads ``df`` from module
    scope; does not take it as an argument.

    Parameters
    ----------
    column_name : str
        Name of the grouping column on the module-level ``df``.
    model_name : str
        Column name of the surprisal-augmented model predictions.
    baseline_model : str, optional
        Column name of the surprisal-free baseline predictions.
        Defaults to ``'baseline'``.

    Returns
    -------
    tuple of (list of float, float)
        Per-group list of ``mean_ll_baseline - mean_ll_model`` and
        the ``std_ll`` of the *last* group's model log-likelihoods.
    """
    difference = []

    for gender in df[column_name].unique():

        data = df[df[column_name]==gender]

        _, mean_ll_1, std_ll_1 = calculate_aic(data['time'], data[baseline_model], 2)
        _, mean_ll_2, std_ll_2 = calculate_aic(data['time'], data[model_name], 3)
        difference.append(mean_ll_1-mean_ll_2)

    return difference, std_ll_2

def calculate_delta_ll_emotion(surprisal, k, emotion_data, std_data):
    """Append per-emotion ``\Delta\log\mathcal{L}`` values for one ``(surprisal, k)``.

    Wraps :func:`akaike_for_column` with ``column_name='emotion'``
    and the conventional ``f"{surprisal} {k} model"`` prediction
    column, then appends the resulting per-emotion
    ``\Delta\log\mathcal{L}`` value into ``emotion_data[e]`` and
    the corresponding std list into ``std_data[e]`` for every
    emotion ``e in 0..4``. Side effect only; nothing is returned.
    On any internal exception (typically a missing model column
    for a degenerate fold), prints a warning and uses zeroes /
    ones instead.

    Parameters
    ----------
    surprisal : str
        Surprisal label (e.g. ``'surprisal GPT'``).
    k : float
        Surprisal-power exponent used to build the model column.
    emotion_data : dict[int, list]
        Per-emotion accumulator for the ``\Delta\log\mathcal{L}``
        sweep.
    std_data : dict[int, list]
        Per-emotion accumulator for the model log-likelihood
        standard-deviation values.

    Returns
    -------
    None
    """
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


def calculate_delta_ll(mode, **kwargs):
    """Dispatcher za calculate_delta_ll varijante u ovom fajlu (P-009).

    Dostupni mode-ovi u Linear regression/results.py:
      - "emotion"  → calculate_delta_ll_emotion(surprisal, k, emotion_data, std_data)
    """
    mapping = {
        "emotion": calculate_delta_ll_emotion,
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return mapping[mode](**kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for mode '{mode}': {e}")


file_path = output_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)
df = df[df['speaker gender']=='m']
df = df[df['word type']!='function']
df = df[df['word type']!='content']

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
    calculate_delta_ll(mode="emotion", surprisal=surprisal_gpt_2, k=k, emotion_data=emotion_data_gpt_2, std_data=gpt_std_2)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_gpt_3, k=k, emotion_data=emotion_data_gpt_3, std_data=gpt_std_3)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_bert, k=k, emotion_data=emotion_data_bert, std_data=bert_std)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_bertic, k=k, emotion_data=emotion_data_bertic, std_data=bertic_std)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_yugo, k=k, emotion_data=emotion_data_yugo, std_data=yugo_std)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_2_alpha4, k=k, emotion_data=emotion_data_ngram_2_alpha4, std_data=ngram_std_2_alpha4)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_3_alpha4, k=k, emotion_data=emotion_data_ngram_3_alpha4, std_data=ngram_std_3_alpha4)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_4_alpha4, k=k, emotion_data=emotion_data_ngram_4_alpha4, std_data=ngram_std_4_alpha4)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_5_alpha4, k=k, emotion_data=emotion_data_ngram_5_alpha4, std_data=ngram_std_5_alpha4)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_2_alpha4, k=k, emotion_data=emotion_data_ngram_2_alpha20, std_data=ngram_std_2_alpha20)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_3_alpha20, k=k, emotion_data=emotion_data_ngram_3_alpha20, std_data=ngram_std_3_alpha20)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_4_alpha20, k=k, emotion_data=emotion_data_ngram_4_alpha20, std_data=ngram_std_4_alpha20)
    calculate_delta_ll(mode="emotion", surprisal=surprisal_ngram_5_alpha20, k=k, emotion_data=emotion_data_ngram_5_alpha20, std_data=ngram_std_5_alpha20)


def plot_data(emotion, emotion_data, std_data, plt_number, c):
  """Scatter ``\Delta\log\mathcal{L}`` versus ``k`` for one emotion.

  Plots the per-emotion sweep ``emotion_data[emotion]``
  (function of ``k`` along the module-level ``x_axis``) as a
  scatter on subplot ``(2, 5, plt_number)`` of the active
  matplotlib figure, and shades a +/- one-std band around it
  using ``np.std(emotion_data[emotion])``. The std list
  ``std_data`` is accepted for API compatibility but is not used
  for the band; the band is the std of the y-axis values
  themselves. Side effect only; nothing is returned.

  Parameters
  ----------
  emotion : int
      Numeric emotion id (``0..4``); index into module-level
      ``emotion_names``.
  emotion_data : dict[int, list]
      Per-emotion ``\Delta\log\mathcal{L}`` accumulator.
  std_data : dict[int, list]
      Per-emotion std accumulator (unused for the band).
  plt_number : int
      1-based subplot index in a 1x5 grid (within a 2-row
      figure).
  c : tuple of float
      RGBA color tuple for the scatter points.

  Returns
  -------
  None
  """

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

emotion_names = ['neutral', 'happy', 'sad', 'scared', 'angry']
fig = plt.figure(figsize=(15,7))

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


# results for paper
fig = plt.figure(figsize=(15,7))

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_ngram_4_alpha4, ngram_std_3_alpha4, emotion + 1, (1, 0, 1, 1))
  plot_data(emotion, emotion_data_gpt_2, gpt_std_2, emotion + 1, (0, 0 , 1, 1))
  plot_data(emotion, emotion_data_yugo, yugo_std, emotion + 1, (1, 0 , 0, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['3-gram','3-gram std', 'gpt-2','gpt-2 std', 'yugo', 'yugo std'])


# results for paper
fig = plt.figure(figsize=(15,7))

for emotion in range(0,5):
  plt.subplot(1, 5, emotion + 1)
  plot_data(emotion, emotion_data_bert, bert_std, emotion + 1, (1, 0, 0, 1))
  plot_data(emotion, emotion_data_bertic, bertic_std, emotion + 1, (0, 0 , 1, 1))

# Adjust the layout to prevent overlapping labels
plt.subplots_adjust(wspace=0.5)  # Adjust the spacing as needed
# Add a common x-axis label
fig.text(0.5, 0.45, '$k$', ha='center', va='center', fontsize=20)
# Add a common y-axis label
fig.text(0.04, 0.70, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=20)
fig.legend(['bert', 'bert std', 'bertic', 'berti std'])

