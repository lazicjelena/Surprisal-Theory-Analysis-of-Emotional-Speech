# -*- coding: utf-8 -*-
"""baseline_model.py

Jelenina skripta
lazic.jelenaa@gmail.com
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math
import os
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def add_column(df, k=0):
    
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

file_path = os.path.join('..','podaci','split-over data', 'general_data.csv') 
df = pd.read_csv(file_path)
df = df.replace("nan", np.nan)
df = df[df['time']!=0]

for k in range(0,5):
    results_df = add_column(df, k)    
    df = pd.merge(df, results_df, how='left')


output_file_path = os.path.join('..','podaci','split-over results', 'baseline_results_data.csv') 
df.to_csv(output_file_path, index=False)