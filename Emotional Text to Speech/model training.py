# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:25:24 2024

@author: Jelena
"""
from sklearn.linear_model import LinearRegression
import pandas as pd 
import os

# baseline model parameters and results    
baseline_results_path = os.path.join('..','podaci','text-to-speech', 'liste', "baseline_results.csv")
baseline_df = pd.read_csv(baseline_results_path)

# surprisal gpt-2 paremetrs
surprisal_path = os.path.join('..','podaci','split-over data', 'Surprisal GPT-2.csv')
data = pd.read_csv(surprisal_path)

# fonteic paremeter
ff_path = os.path.join('..','podaci', 'information measurements parameters', "fonetic_features1.csv") 
ff_data = pd.read_csv(ff_path)
ff_data = ff_data.drop('sentence', axis=1)
ff_data = ff_data.drop_duplicates()

fonem_list = ['vokali', 'alveolarni', 'palatalni', 'bilabijalni', 'labio_dentalni',
               'labialni', 'zubni', 'palatalni', 'zadnjonepcani', 'zvucni', 'bezvucni']

# Add new column for information_values
iv_df_path = os.path.join('..','podaci', 'information measurements parameters', "information_value.csv") 
iv_data = pd.read_csv(iv_df_path)
iv_data = iv_data.groupby('Word', as_index=False).agg({'Non-context Embedding': 'mean'})
iv_data = iv_data[['Word', 'Non-context Embedding']]
iv_data.rename(columns={'Word': 'word'}, inplace=True)

def add_column_with_surprisal(df, surprisal, fonem_list, emotion):
    
    columns = df.columns.tolist()
    training_columns = ['length', 'log probability', 'Non-context Embedding', surprisal] + fonem_list
    
    for i in range(1,4):
        training_columns.append(f"length -{i}")
        training_columns.append(f"log probability -{i}")
        training_columns.append(f"{surprisal} -{i}")
        training_columns.append(f"Non-context Embedding -{i}")
            
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


df = data
df = pd.merge(df, iv_data, how='left')

column = 'Non-context Embedding'
df[f"{column} -1"] = df[f"{column}"].shift(1)
#df.loc[df['target sentence'] != df['target sentence'].shift(1), [f"{column} -1"]] = pd.NA
df[f"{column} -2"] = df[f"{column} -1"].shift(1)
#df.loc[df['target sentence'] != df['target sentence'].shift(1), [f"{column} -2"]] = pd.NA
df[f"{column} -3"] = df[f"{column} -2"].shift(1)
#df.loc[df['target sentence'] != df['target sentence'].shift(1), [f"{column} -3"]] = pd.NA

df = pd.merge(df, ff_data, how='left')

df = pd.merge(df, baseline_df, how='left')

for emotion in [0,1,2,3,4]:
    
    emotion_data = df[df['emotion'] == emotion]    
    results_df = add_column_with_surprisal(emotion_data, 'Surprisal GPT-2', fonem_list, emotion)
    df = pd.merge(df, results_df, how='left')

columns_to_merge = [f"Surprisal GPT-2 {emotion} model" for emotion in range(0,5)]
df["Surprisal GPT-2 model"] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
df = df.drop(columns=columns_to_merge)
    
results_path = os.path.join('..','podaci','text-to-speech', 'liste', "Surprisal GPT-2_results.csv")
df.to_csv(results_path, index=False)