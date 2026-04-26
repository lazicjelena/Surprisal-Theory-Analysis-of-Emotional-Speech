# -*- coding: utf-8 -*-
"""baseline_model.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ovdje se dobijaju rezultati za model koji ne uzima u obzir surprisale.

Pipeline role
-------------
Trains the surprisal-free baseline duration regression for the
``Duration Prediction based on Surprisals/`` analysis on
``../podaci/training data/general_data.csv`` (built by
``transform_data_into_dataframe.py``). Rows with ``time == 0``
are dropped; per fold a :class:`LinearRegression` is fit on
``log2(time)`` (with a 3-sigma outlier filter on the training
side) using ``length`` and ``log probability``, and out-of-fold
predictions are placed in a ``baseline`` column. The merged
result is written to
``../podaci/results - linear regression/baseline_results_data.csv``
and serves as the reference column subtracted in
:math:`\Delta\log\mathcal{L}` calculations performed by
``surprisal_results.py``.
"""


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import os

file_path = os.path.join('..','podaci','training data', 'general_data.csv') 
df = pd.read_csv(file_path)
df = df[df['time']!=0]

mse_list = []
results_df = pd.DataFrame(columns = df.columns.tolist().append('baseline'))

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
    model.fit(train_data[['length', 'log probability']], y_train)

    y_pred = model.predict(test_data[['length', 'log probability']])

    test_data.loc[:, 'baseline'] = y_pred
    # Concatenate the DataFrames along rows (axis=0)
    results_df = pd.concat([results_df, test_data], axis=0)
    
    y_test['time'] = y_test['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)


# Calculate the average of mse_list
average_mse = sum(mse_list) / len(mse_list)

for f, mse in zip(df['fold'].unique(),mse_list):
  print(f"Fold: {f}  MSE: {mse}")
print(f"Average mse over folds: {average_mse}")

results_df = pd.merge(df, results_df, how='left')
output_file_path = os.path.join('..','podaci','results - linear regression', 'baseline_results_data.csv') 
results_df.to_csv(output_file_path, index=False)