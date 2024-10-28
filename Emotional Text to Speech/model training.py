# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:25:24 2024

@author: Jelena
"""
from sklearn.linear_model import LinearRegression
import pandas as pd 
import os
    
baseline_results_path = os.path.join('..','podaci','text-to-speech', 'liste', "baseline_results.csv")
baseline_df = pd.read_csv(baseline_results_path)
file_path = os.path.join('..','podaci','split-over data')
surprisal_column_name = ['Surprisal GPT-2'
                         #'Surprisal Yugo', 
                         #'Surprisal ngram-3',
                         #'Surprisal BERT',
                         #'Surprisal BERTic'
                         ]

def add_column_with_surprisal(df, surprisal, emotion):
    
    columns = df.columns.tolist()
    training_columns = ['length', 'log probability', surprisal]
    
    for i in range(1,3):
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
        #y_train['time'] = y_train['time'].apply(lambda x: math.log2(x) if x > 0 else float('nan'))
        
        # reduce outliers
        gaussian_condition = (y_train['time'] - y_train['time'].mean()) / y_train['time'].std() < 3
        train_data = train_data[gaussian_condition]
        y_train = y_train[gaussian_condition]
        
        model = LinearRegression()
        model.fit(train_data[training_columns], y_train)
        
        y_pred = model.predict(test_data[training_columns])
        test_data.loc[:, f"{surprisal} {emotion} model"] = y_pred
            
        # Concatenate the DataFrames along rows (axis=0)
        results_df = pd.concat([results_df, test_data], axis=0)
        
   
    return results_df.drop_duplicates()

for surprisal in surprisal_column_name:
    
    df_path = os.path.join(file_path, surprisal + '.csv') 
    df = pd.read_csv(df_path)
    #df = df[df['speaker'].isin(range(1, 5))]
    df = df[df['speaker']==1]
    df = pd.merge(df, baseline_df, how='left')
    
    for emotion in [0,1,2,3,4]:
        emotion_data = df[df['emotion'] == emotion]
        
        results_df = add_column_with_surprisal(emotion_data, surprisal, emotion)
        df = pd.merge(df, results_df, how='left')

    columns_to_merge = [ f"{surprisal} {emotion} model" for emotion in range(0,5)]
    df[f"{surprisal} model"] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns=columns_to_merge)
    
    results_path = os.path.join('..','podaci','text-to-speech', 'liste', f"{surprisal}_results.csv")
    df.to_csv(results_path, index=False)