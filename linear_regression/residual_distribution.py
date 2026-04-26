# -*- coding: utf-8 -*-
"""residual_distribution.py
Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta predstavlja reziduale predikcije izgovora. Dobijaju se neznatni rezultati
koji nisu koristeni u radu.

Pipeline role
-------------
Diagnostic plotting script (off the production critical path).
For male speakers in the master training table
``../podaci/training_data.csv``, fits the same per-fold
log-time linear regression as ``final_graphs.py``
(:func:`inf_k_model`) at four hand-picked surprisal-power
exponents -- ``GPT k=1.75``, ``BERT k=0.25``, ``ngram3 k=1.75``,
``Yugo k=1.75`` -- and overlays per-emotion KDE plots of the
residuals (model prediction minus true ``time``) against the
baseline residuals via :func:`plot_residuals`. Used during
exploratory analysis only; the resulting figures were not
included in the thesis.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import math 
import seaborn as sns

def inf_k_model(df, k, surprisal):
    """Fit ``log2(time) ~ length + log_prob + surprisal**k`` per fold.

    Identical in spirit to :func:`final_graphs.inf_k_model` but the
    power-transformed surprisal column is *not* dropped from the
    returned DataFrame, so it remains available downstream for
    diagnostic plotting. Adds ``f"{surprisal} {k}"`` (the
    transformed feature) and ``f"{surprisal} {k} model"`` (the
    fold-out predictions) to ``df`` and returns the concatenation
    of all folds. Outliers on the training side are dropped with a
    3-sigma rule on ``log2(time)``.

    Parameters
    ----------
    df : pandas.DataFrame
        Master training table; must contain ``length``,
        ``log probability``, ``time``, ``fold``, and the named
        ``surprisal`` column.
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


file_path = output_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)
df = df[df['speaker gender']=='m']

import warnings
# Filter out SettingWithCopyWarning
warnings.filterwarnings("ignore")
surprisal_gpt_2 = 'surprisal GPT'
surprisal_gpt_3 = 'surprisal GPT3'
surprisal_bert = 'surprisal BERT'
surprisal_ngram_2 = 'surprisal ngram2'
surprisal_ngram_3 = 'surprisal ngram3'
surprisal_ngram_4 = 'surprisal ngram4'
surprisal_ngram_5 = 'surprisal ngram5'
surprisal_yugo = 'surprisal yugo'


df = inf_k_model(df, 1.75, surprisal_gpt_2)
df = inf_k_model(df, 0.25, surprisal_bert)
df = inf_k_model(df, 1.75, surprisal_ngram_3)
df = inf_k_model(df, 1.75, surprisal_yugo)

# Reset warnings to default behavior (optional)
warnings.resetwarnings()


def plot_residuals(emotion, model, df, plt_number):
    """KDE plot of model vs. baseline residuals for one emotion.

    Slices ``df`` to rows with ``emotion == emotion``, computes the
    residual ``df[model] - df['time']`` and the baseline residual
    ``df['baseline'] - df['time']``, and plots both as KDE curves
    on subplot ``(2, 5, plt_number)`` of the active matplotlib
    figure. Side effect only; nothing is returned.

    Parameters
    ----------
    emotion : int
        Numeric emotion id (``0..4``); index into module-level
        ``emotion_names``.
    model : str
        Name of the model-prediction column in ``df``.
    df : pandas.DataFrame
        Table holding ``emotion``, ``time``, ``baseline``, and
        the named ``model`` column.
    plt_number : int
        1-based subplot index in a 2x5 grid.

    Returns
    -------
    None
    """
    # Plot results for the specified emotion and model
    data = df[df['emotion']==emotion]
    plt.subplot(2, 5, plt_number)

    # Plot KDE plot for baseline
    sns.kdeplot(data['baseline'] - data['time'], color='blue', linewidth=2)

    # Plot KDE plot for the model
    y = data[model] - data['time'] 
    sns.kdeplot(y, color='red', linewidth=2)

    plt.title(emotion_names[emotion], fontsize=20)

    return

fig = plt.figure(figsize=(15, 7))
emotion_names = ['неутрално', 'срећно', 'тужно', 'уплашено', 'љуто']
model = 'surprisal BERT 0.25 model'

for emotion in range(0, 5):
    plot_residuals(emotion, model, df, emotion + 1)

fig.legend(['baseline', 'bert'])



fig = plt.figure(figsize=(15, 7))
emotion_names = ['неутрално', 'срећно', 'тужно', 'уплашено', 'љуто']
model = 'surprisal GPT 1.75 model'

for emotion in range(0, 5):
    plot_residuals(emotion, model, df, emotion + 1)

fig.legend(['baseline', 'bert'])