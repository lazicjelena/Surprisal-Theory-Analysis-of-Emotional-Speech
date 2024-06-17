# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:49:47 2024

@author: Jelena
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


# Read surprisal data
data_paths = []
data_paths.append(os.path.join('..','podaci','word_surprisal_ngram3_alpha20.csv'))
data_paths.append(os.path.join('..','podaci','word_surprisals_gpt2.csv'))
data_paths.append(os.path.join('..','podaci','word_surprisals_yugo.csv'))
data_paths.append(os.path.join('..','podaci','word_surprisals_bert.csv'))
data_paths.append(os.path.join('..','podaci','word_surprisals_bertic.csv'))

data = pd.DataFrame(columns = ['Sentence', 'Word'])
for path in data_paths:
    new_data = pd.read_csv(path)
    data = pd.merge(new_data, data, how='left', on=['Sentence','Word'])
    
    
# Add unigram frequencies
unigram_freq_path = os.path.join('..','podaci','wordlist_frequencies.csv') 
new_data = pd.read_csv(unigram_freq_path)
data = pd.merge(new_data[['Sentence', 'Word', 'Log Probability']], data, how='left', on=['Sentence','Word'])
    
# Define a function to normalize the columns
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())


# Ensure 'Word' column is treated as categorical
data['Word'] = pd.Categorical(data['Word'], ordered=True)

# Plot each sentence on separate plots
for sentence_id, group in data.groupby('Sentence'):
    plt.figure(figsize=(10, 6))
   # plt.plot(group['Word'], group['Log Probability'], label='-log(f)')
   # plt.plot(group['Word'], group['Surprisal ngram-3'])
    plt.plot(group['Word'], group['Surprisal GPT-2'])
    plt.plot(group['Word'], group['Surprisal Yugo'])
  #  plt.plot(group['Word'], group['Surprisal BERT'])
  #  plt.plot(group['Word'], group['Surprisal BERTic'])
    
    
   # plt.scatter(group['Word'], group['Surprisal ngram-3'], label='3-gram' , marker='o', s=80)
    plt.scatter(group['Word'], group['Surprisal GPT-2'], label='GPT-2' , marker='o', s=80)
    plt.scatter(group['Word'], group['Surprisal Yugo'], label='Yugo-GPT' , marker='o', s=80)
   # plt.scatter(group['Word'], group['Surprisal BERT'], label='BERT', marker='o', s=80)
   # plt.scatter(group['Word'], group['Surprisal BERTic'], label='BERTic', marker='o', s=80)
    
    #plt.title(f'Sentence {sentence_id}', fontsize=20)
    plt.xlabel('word', fontsize=20)
    plt.ylabel('surprisal', fontsize=20)
    plt.xticks(rotation=45, fontsize=15)  # Set xticks fontsize and rotation
    plt.legend(fontsize=15)
    plt.tight_layout()

    plt.show()
    

