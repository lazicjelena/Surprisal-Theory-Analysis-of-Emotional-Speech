# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta sluzi za objedinjavanje podata svih govornika i njihovu raspodjelo u
foldovima. 

"""

import os
import pandas as pd

def add_word_type(data, freq_df, column_name):
    
    word_type_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        
        words = row['word'].split(' ')
        sentence = row['target sentence']
        
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
          
        word_type_value = ''
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq_s = freq_df[freq_df['Sentence'] == sentence]
            freq = freq_s[freq_s['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            if not freq.empty:
                word_type_value += ' '
                word_type_value += freq[column_name].values[0 + list_of_words.count(word)]
            else:
              word_type_value += ' '
              print('error')
              print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s):
              list_of_words = []

        word_type_list.append(word_type_value.strip())

    return word_type_list

def lookup_freq(data, freq_df, column_name):
    
    log_prob_list = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        
        words = row['word'].split(' ')
        log_probability_value = 0
        
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq = freq_df[freq_df['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            if not freq.empty:
                log_probability_value += freq[column_name].values[0]
            else:
              log_probability_value += 0
              print('error')
              print(word)

        log_prob_list.append(log_probability_value)

    return log_prob_list

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

data.rename(columns={'taget sentence': 'target sentence'}, inplace=True)

# read frequencies
f_path = os.path.join('..','podaci', 'wordlist_frequencies.csv') 
freq_df = pd.read_csv(f_path)
log_probab_list = lookup_freq(data, freq_df, 'Log Probability')
data['log probability'] = log_probab_list

word_type_path = os.path.join('..','podaci', 'word_type.csv') 
word_type_df = pd.read_csv(word_type_path)
word_type_list = add_word_type(data, word_type_df, 'Type')
data['word type'] = word_type_list

folds_path =   os.path.join('..','podaci', 'folds.csv') 
folds_df = pd.read_csv(folds_path)
# Merge the two DataFrames on 'target sentence'
merged_df = pd.merge(data, folds_df, on='target sentence', how='left')

# Save the concatenated data to a CSV file
output_csv_path = os.path.join('..','podaci','training data', 'general_data.csv') 
merged_df.to_csv(output_csv_path, index=False)
