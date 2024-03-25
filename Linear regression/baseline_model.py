# -*- coding: utf-8 -*-
"""baseline_model.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ovdje se dobijaju rezultati za model koji ne uzima u obzir surprisale.
"""


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import math

file_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)

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

    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    test_data.loc[:, 'baseline'] = y_pred
    # Concatenate the DataFrames along rows (axis=0)
    results_df = pd.concat([results_df, test_data], axis=0)


# Calculate the average of mse_list
average_mse = sum(mse_list) / len(mse_list)

for f, mse in zip(df['fold'].unique(),mse_list):
  print(f"Fold: {f}  MSE: {mse}")
print(f"Average mse over folds: {average_mse}")

results_df

results_df.to_csv(file_path, index=False)