# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:33:04 2024

@author: Jelena
"""

import pandas as pd
import os


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
            surprisal_s = surprisal_df[surprisal_df['target sentence'] == sentence]
            surprisal_w = surprisal_s[surprisal_s['word'] == word]

            # Extract the 'Surprisal' value for the filtered word
            if len(surprisal_w)>0:
                surprisal_value += surprisal_w[column_name].values[0 + list_of_words.count(word)]
            else:
                surprisal_value += 0
               # print('error')
               # print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(surprisal_s):
              list_of_words = []

        surprisal_list.append(surprisal_value)

    return surprisal_list


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

# read prominence data
prominence_path = os.path.join('..','podaci', 'correlation data','prominence_data.csv') 
prominence_df = pd.read_csv(prominence_path)

for surprisal_csv_file,  column_name in zip(surprisal_df_list, surprisal_name_list):
    
    surprisal_path = os.path.join('..','podaci', 'correlation data', "former_" + surprisal_csv_file) 
    surprisal_df = pd.read_csv(surprisal_path)

    prominence_df[column_name] = lookup_features(prominence_df, surprisal_df, column_name)

    for k in range(1,11):
        prominence_df[column_name + f" k={k}"] = lookup_features(prominence_df, surprisal_df, column_name + f" k={k}")


# Save the concatenated data to a CSV file
output_csv_path = os.path.join('..','podaci', 'correlation data', "full_former_surprisal_data.csv") 
prominence_df.to_csv(output_csv_path, index=False)