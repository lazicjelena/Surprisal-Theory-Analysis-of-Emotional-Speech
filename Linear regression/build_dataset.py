# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta sluzi za objedinjavanje podata svih govornika i njihovu raspodjelo u
foldovima. 
Mora da se pokrece svaki put kad se doda novo obiljezje.
"""

import os
import pandas as pd

def lookup_features(data, freq_df, column_name):
    log_prob_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        words = row['word'].split(' ')
        sentence = row['target sentence']
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
        #print(index)
        log_probability_value = 0
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq_s = freq_df[freq_df['Sentence'] == sentence]
            freq = freq_s[freq_s['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            if not freq.empty:
                log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
            else:
              log_probability_value += 0
              print('error')
              print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s):
              list_of_words = []

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


# Define the base directory
file_path = os.path.join('..','podaci', 'concatenated_data.csv') 
data = pd.read_csv(file_path)


# read frequencies
f_path = os.path.join('..','podaci', 'wordlist_frequencies.csv') 
freq_df = pd.read_csv(f_path)

log_probab_list = lookup_features(data, freq_df, 'Log Probability')
data['log probability'] = log_probab_list

# yugo model
yugo_path = os.path.join('..','podaci', 'word_surprisals_yugo.csv') 
surprisal_yugo = pd.read_csv(yugo_path)
surprisal_yugo_list = lookup_features(data, surprisal_yugo, 'Surprisal Yugo')
data['surprisal yugo'] = surprisal_yugo_list


# gpt model
gpt_path = os.path.join('..','podaci', 'word_surprisals_gpt2.csv') 
surprisal_gpt = pd.read_csv(gpt_path)
gpt3_path = os.path.join('..','podaci', 'word_surprisals_gpt3.csv') 
surprisal_gpt3 = pd.read_csv(gpt3_path)

surprisal_gpt_list = lookup_features(data, surprisal_gpt, 'Surprisal GPT-2')
data['surprisal GPT'] = surprisal_gpt_list
surprisal_gpt3_list = lookup_features(data, surprisal_gpt3, 'Surprisal GPT-3')
data['surprisal GPT3'] = surprisal_gpt3_list

# bert model
bert_path = os.path.join('..','podaci', 'word_surprisals_bert.csv') 
surprisal_bert = pd.read_csv(bert_path)

surprisal_bert_list = lookup_features(data, surprisal_bert, 'Surprisal BERT')
data['surprisal BERT'] = surprisal_bert_list

bertic_path = os.path.join('..','podaci', 'word_surprisals_bertic.csv') 
surprisal_bertic = pd.read_csv(bertic_path)

surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal BERTic')
data['surprisal BERTic'] = surprisal_bertic_list

# ngram models
ngram2_path = os.path.join('..','podaci', 'word_surprisal_ngram2_alpha4.csv') 
surprisal_ngram2 = pd.read_csv(ngram2_path)
surprisal_ngram2_list = lookup_features(data, surprisal_ngram2, 'Surprisal ngram-3')
data['surprisal ngram2 alpha4'] = surprisal_ngram2_list

ngram2_path = os.path.join('..','podaci', 'word_surprisal_ngram2_alpha20.csv') 
surprisal_ngram2 = pd.read_csv(ngram2_path)
surprisal_ngram2_list = lookup_features(data, surprisal_ngram2, 'Surprisal ngram-3')
data['surprisal ngram2 alpha20'] = surprisal_ngram2_list

ngram3_path = os.path.join('..','podaci', 'word_surprisal_ngram3_alpha4.csv') 
surprisal_ngram3 = pd.read_csv(ngram3_path)
surprisal_ngram3_list = lookup_features(data, surprisal_ngram3, 'Surprisal ngram-3')
data['surprisal ngram3 alpha4'] = surprisal_ngram3_list

ngram3_path = os.path.join('..','podaci', 'word_surprisal_ngram3_alpha20.csv') 
surprisal_ngram3 = pd.read_csv(ngram3_path)
surprisal_ngram3_list = lookup_features(data, surprisal_ngram3, 'Surprisal ngram-3')
data['surprisal ngram3 alpha20'] = surprisal_ngram3_list

ngram4_path = os.path.join('..','podaci', 'word_surprisal_ngram4_alpha4.csv') 
surprisal_ngram4 = pd.read_csv(ngram4_path)
surprisal_ngram4_list = lookup_features(data, surprisal_ngram4, 'Surprisal ngram-3')
data['surprisal ngram4 alpha4'] = surprisal_ngram4_list

ngram4_path = os.path.join('..','podaci', 'word_surprisal_ngram4_alpha20.csv') 
surprisal_ngram4 = pd.read_csv(ngram4_path)
surprisal_ngram4_list = lookup_features(data, surprisal_ngram4, 'Surprisal ngram-3')
data['surprisal ngram4 alpha20'] = surprisal_ngram4_list

ngram5_path = os.path.join('..','podaci', 'word_surprisal_ngram5_alpha4.csv') 
surprisal_ngram5 = pd.read_csv(ngram5_path)
surprisal_ngram5_list = lookup_features(data, surprisal_ngram5, 'Surprisal ngram-3')
data['surprisal ngram5 alpha4'] = surprisal_ngram5_list

ngram5_path = os.path.join('..','podaci', 'word_surprisal_ngram5_alpha20.csv') 
surprisal_ngram5 = pd.read_csv(ngram5_path)
surprisal_ngram5_list = lookup_features(data, surprisal_ngram5, 'Surprisal ngram-3')
data['surprisal ngram5 alpha20'] = surprisal_ngram5_list

folds_path =   os.path.join('..','podaci', 'folds.csv') 
folds_df = pd.read_csv(folds_path)

# Merge the two DataFrames on 'target sentence'
merged_df = pd.merge(data, folds_df, on='target sentence', how='left')
merged_df = merged_df[merged_df['time']!=0]

# Save the concatenated data to a CSV file
output_csv_path = os.path.join('..','podaci', 'training_data.csv') 
merged_df.to_csv(output_csv_path, index=False)