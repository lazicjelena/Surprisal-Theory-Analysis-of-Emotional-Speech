# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:02:47 2025

@author: Jelena
"""

from my_functions import lookup_features, add_word_type
import pandas as pd
import os


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

data = data.rename(columns={'taget sentence':'target sentence'})

# read frequencies
f_path = os.path.join('..','podaci', 'wordlist_frequencies.csv') 
freq_df = pd.read_csv(f_path)

log_probability_list = []
for _,row in data.iterrows():
    
    words = row['word'].lower().strip()
    lp_value = 0
    for word in words.split(' '):
        if word =='dogovrili':
            word ='dogovorili'
        if word =='pet':
            word = '5'
        lp_value += freq_df[freq_df['Word'] == word]['Log Probability'].iloc[0]
        
    log_probability_list.append(lp_value)
        
data['log probability'] = log_probability_list


# yugo model
yugo_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_yugo.csv') 
surprisal_yugo = pd.read_csv(yugo_path)
surprisal_yugo_list = lookup_features(data, surprisal_yugo, 'Surprisal Yugo')
data['surprisal yugo'] = surprisal_yugo_list


# gpt model
gpt_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_gpt2.csv') 
surprisal_gpt = pd.read_csv(gpt_path)
surprisal_gpt_list = lookup_features(data, surprisal_gpt, 'Surprisal GPT-2')
data['surprisal GPT'] = surprisal_gpt_list


# bert model
bert_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bert.csv') 
surprisal_bert = pd.read_csv(bert_path)
surprisal_bert_list = lookup_features(data, surprisal_bert, 'Surprisal BERT')
data['surprisal BERT'] = surprisal_bert_list

bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bertic.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal BERTic')
data['surprisal BERTic'] = surprisal_bertic_list

bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_roberta.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal RoBERTa')
data['surprisal RoBERTa'] = surprisal_bertic_list

bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_albert.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal ABERT')
data['surprisal ALBERT'] = surprisal_bertic_list

# bert model uni
bert_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bert_uni.csv') 
surprisal_bert = pd.read_csv(bert_path)
surprisal_bert_list = lookup_features(data, surprisal_bert, 'Surprisal BERT uni')
data['surprisal BERT uni'] = surprisal_bert_list

bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bertic_uni.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal BERTic uni')
data['surprisal BERTic uni'] = surprisal_bertic_list

bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_roberta_uni.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal RoBERTa uni')
data['surprisal RoBERTa uni'] = surprisal_bertic_list

bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_albert_uni.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal ALBERT uni')
data['surprisal ALBERT uni'] = surprisal_bertic_list

# ngram models
ngram3_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisal_ngram3_alpha4.csv') 
surprisal_ngram3 = pd.read_csv(ngram3_path)
surprisal_ngram3_list = lookup_features(data, surprisal_ngram3, 'Surprisal ngram-3')
data['surprisal ngram3 alpha4'] = surprisal_ngram3_list


word_type_path = os.path.join('..','podaci', 'word_type.csv') 
word_type_df = pd.read_csv(word_type_path)
word_type_list = add_word_type(data, word_type_df, 'Type')
data['word type'] = word_type_list

folds_path =   os.path.join('..','podaci', 'folds.csv') 
folds_df = pd.read_csv(folds_path)

# Merge the two DataFrames on 'target sentence'
merged_df = pd.merge(data, folds_df, on='target sentence', how='left')
merged_df = merged_df[merged_df['time']!=0]

# Save the concatenated data to a CSV file
output_csv_path = os.path.join('..','podaci', 'training_data.csv') 
merged_df.to_csv(output_csv_path, index=False)