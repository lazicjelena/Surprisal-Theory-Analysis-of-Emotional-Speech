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
#data_paths.append(os.path.join('..','podaci','word_surprisal_ngram3_alpha20.csv'))
#data_paths.append(os.path.join('..','podaci','word_surprisals_gpt2.csv'))
#data_paths.append(os.path.join('..','podaci','word_surprisals_yugo.csv'))
data_paths.append(os.path.join('..','podaci','word_surprisals_bert.csv'))
data_paths.append(os.path.join('..','podaci','word_surprisals_bertic.csv'))

data = pd.DataFrame(columns = ['Sentence', 'Word'])
for path in data_paths:
    new_data = pd.read_csv(path)
    data = pd.merge(new_data, data, how='left', on=['Sentence','Word'])
    
# Add unigram frequencies
unigram_freq_path = os.path.join('..','podaci','wordlist_frequencies.csv') 
new_data = pd.read_csv(unigram_freq_path)
data = pd.merge(new_data[[ 'Word', 'Log Probability']], data, how='left', on=['Word'])

target_sentences_path = os.path.join('..','podaci','target_sentences.csv') 
target_sentences_df = pd.read_csv(target_sentences_path)    

# Define a function to normalize the columns
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

# Plot each sentence on separate plots
for sentence_id in data['Sentence'].unique():
    plt.figure(figsize=(10, 6))
    try:
        group = data[data['Sentence'] == sentence_id]
        sentence = target_sentences_df['Text'][sentence_id].split()
        
        surprisal_values = []
        for word in sentence:
            #surprisal_value = group[group['Word']==word]['Surprisal ngram-3'].values[0]
            #surprisal_value = group[group['Word']==word]['Surprisal GPT-2'].values[0]
            #surprisal_value = group[group['Word']==word]['Surprisal BERT'].values[0]
            surprisal_values.append(surprisal_value)
        plt.plot(sentence, surprisal_values, linewidth = 3)
        
        surprisal_values_1 = []
        for word in sentence:
            #surprisal_value = group[group['Word']==word]['Surprisal Yugo'].values[0]
            surprisal_value = group[group['Word']==word]['Surprisal BERTic'].values[0]
            surprisal_values_1.append(surprisal_value)
        plt.plot(sentence, surprisal_values_1, linewidth = 3)
        
       # plt.scatter(sentence,surprisal_values, label='3-gram' , marker='o', s=80)
       # plt.scatter(sentence, surprisal_values, label='GPT-2' , marker='o', s=120)
       # plt.scatter(sentence, surprisal_values_1, label='Yugo-GPT' , marker='o', s=120)
        plt.scatter(sentence, surprisal_values, label='BERT', marker='o', s=80)
        plt.scatter(sentence, surprisal_values_1, label='BERTic', marker='o', s=80)
       
        plt.xlabel('word', fontsize=20)
        plt.ylabel('surprisal', fontsize=20)
        plt.xticks(rotation=45, fontsize=20)  # Set xticks fontsize and rotation
        plt.legend(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Surprisal Values in Example Sentence', fontsize= 25)
        plt.tight_layout()
    
        plt.show()
    except:
        print(f"Error while processing sentence: {sentence_id}")
            
    
# Plot each sentence on separate plots
for sentence_id in data['Sentence'].unique():
    plt.figure(figsize=(10, 6))
    try:
        group = data[data['Sentence'] == sentence_id]
        sentence = target_sentences_df['Text'][sentence_id].split()
        
        surprisal_values = []
        forword in sentence:
            #surprisal_value = group[group['Word']==word]['Surprisal ngram-3'].values[0]
            #surprisal_value = group[group['Word']==word]['Surprisal GPT-2'].values[0]
            surprisal_value = group[group['Word']==word]['Surprisal BERT'].values[0]
            surprisal_values.append(surprisal_value)
        plt.plot(sentence, surprisal_values, linewidth = 3)
        
        surprisal_values_1 = []
        for word in sentence:
            #surprisal_value = group[group['Word']==word]['Surprisal Yugo'].values[0]
            surprisal_value = group[group['Word']==word]['Surprisal BERTic'].values[0]
            surprisal_values_1.append(surprisal_value)
        plt.plot(sentence, surprisal_values_1, linewidth = 3)
        
       # plt.scatter(sentence,surprisal_values, label='3-gram' , marker='o', s=80)
       # plt.scatter(sentence, surprisal_values, label='GPT-2' , marker='o', s=120)
       # plt.scatter(sentence, surprisal_values_1, label='Yugo-GPT' , marker='o', s=120)
        plt.scatter(sentence, surprisal_values, label='BERT', marker='o', s=80)
        plt.scatter(sentence, surprisal_values_1, label='BERTic', marker='o', s=80)
    
        plt.xlabel('ријечи', fontsize=20)
        plt.ylabel('сурприсал', fontsize=20)
        plt.xticks(rotation=45, fontsize=20)  # Set xticks fontsize and rotation
        plt.legend(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Сурприсал вриједности у примјер реченици', fontsize= 25)
        plt.tight_layout()
    
        plt.show()
    except:
        print(f"Error while processing sentence: {sentence_id}")
    

