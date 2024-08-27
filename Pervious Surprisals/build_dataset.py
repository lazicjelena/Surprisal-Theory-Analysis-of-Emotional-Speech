# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta sluzi za objedinjavanje podata svih govornika i njihovu raspodjelo u
foldovima i surprisala od trenutne i prethodnih rijeci.
"""

import os
import pandas as pd
import numpy as np


def lookup_features(data, surprisal_df, column_name):
    surprisal_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        words = row['word'].split(' ')
        sentence = row['target sentence']
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
          
        surprisal_value = 0
        for word in words:
            # Filter freq_df based on the 'Word' column
            surprisal_s = surprisal_df[surprisal_df['Sentence'] == sentence]
            surprisal_w = surprisal_s[surprisal_s['Word'] == word]

            # Extract the 'Surprisal' value for the filtered word
            if len(surprisal_w)>0:
                surprisal_value += surprisal_w[column_name].values[0 + list_of_words.count(word)]
            else:
              surprisal_value += 0
              print('error')
              print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(surprisal_s):
              list_of_words = []

        surprisal_list.append(surprisal_value)

    return surprisal_list

def calculate_former_surprisal(data, surprisal_name):
    list_of_former_surprisal = []
    current_sentence = 632785931
    
    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        sentence = row['target sentence']
        
        if sentence != current_sentence:
          current_sentence = sentence
          former_surprisal = np.NaN
        
        list_of_former_surprisal.append(former_surprisal)
        former_surprisal = row[surprisal_name]

    return list_of_former_surprisal

# Define the base directory
base_path = os.path.join('..','podaci', 'data') 

# Create an empty DataFrame to store concatenated data
data = pd.DataFrame()

# Loop through top-level folders
for top_folder_name in os.listdir(base_path):
    top_folder_path = os.path.join(base_path, top_folder_name)

    # Check if the path is a directory
    if os.path.isdir(top_folder_path):
        # Loop through subfolders in each top-level folder
        for sub_folder_name in os.listdir(top_folder_path):
            sub_folder_path = os.path.join(top_folder_path, sub_folder_name)

            # Check if the path is a directory
            if os.path.isdir(sub_folder_path):
                # Loop through all files in the subfolder
                for file_name in os.listdir(sub_folder_path):
                    file_path = os.path.join(sub_folder_path, file_name)

                    # Check if the file is a CSV file
                    if file_name.endswith('.csv'):
                        # Read the CSV file and concatenate it to the data DataFrame
                        df = pd.read_csv(file_path)
                        data = pd.concat([data, df], ignore_index=True)

data = data.rename(columns={"taget sentence": "target sentence"})

# read frequencies
f_path = os.path.join('..','podaci', 'wordlist_frequencies.csv') 
freq_df = pd.read_csv(f_path)
freq_df = freq_df.rename(columns={"Word": "word",
                                  "Log Probability": "log probability"})
freq_df = freq_df[["word", "log probability"]]
log_prob_list = []
for index, row in data.iterrows():
    words = row['word'].split(' ')
        
    freq_value = 0
    for word in words:
        freq_value += freq_df[freq_df['word'] == word]['log probability'].values[0]
        
    log_prob_list.append(freq_value)
        
data['log probability'] = log_prob_list

        
# add fold info        
folds_path =   os.path.join('..','podaci', 'folds.csv') 
folds_df = pd.read_csv(folds_path)
data = pd.merge(data, folds_df, on='target sentence', how='left')

# surprisal model
surprisal_df_list = ['word_surprisals_yugo.csv',
                     'word_surprisals_gpt2.csv',
                     'word_surprisals_bert.csv',
                     'word_surprisals_bertic.csv',
                     'word_surprisal_ngram2_alpha4.csv']

surprisal_name_list = ['Surprisal Yugo',
                       'Surprisal GPT-2',
                       'Surprisal BERT',
                       'Surprisal BERTic',
                       'Surprisal ngram-3']

for surprisal_df_path, surprisal_name in zip(surprisal_df_list, surprisal_name_list):
    
    df = data
    data_path = os.path.join('..','podaci', surprisal_df_path) 
    surprisal_data = pd.read_csv(data_path)
    surprisal_values_list = lookup_features(df, surprisal_data, surprisal_name)
    df[surprisal_name] = surprisal_values_list 

    df[surprisal_name + " k=1"] = calculate_former_surprisal(df, surprisal_name)
    print(surprisal_name)
    non_nan = len(df) - df[surprisal_name + " k=1"].isna().sum()
    print(f"1: {non_nan}")
    
    for k in range(2,11):
        df[f"{surprisal_name} k={k}"]= calculate_former_surprisal(df, f"{surprisal_name} k={k-1}")
        non_nan = len(df) - df[f"{surprisal_name} k={k}"].isna().sum()
        print(f"{k}: {non_nan}")

    # Save the concatenated data to a CSV file
    output_csv_path = os.path.join('..','podaci', 'correlation data', "former_" + surprisal_df_path) 
    df.to_csv(output_csv_path, index=False)
