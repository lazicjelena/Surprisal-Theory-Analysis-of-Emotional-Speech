# -*- coding: utf-8 -*-
"""residual_distribution.py
Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta predstavlja reziduale predikcije izgovora. Dobijaju se neznatni rezultati
koji nisu koristeni u radu.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import math 
import seaborn as sns

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