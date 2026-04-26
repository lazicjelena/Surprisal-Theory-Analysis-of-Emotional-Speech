# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:49:47 2024

@author: Jelena

Pipeline role
-------------
Per-sentence surprisal visualisation. Loads the five per-word
surprisal CSVs produced by the surprisal-estimation stage
(``../podaci/surprisal values/word_surprisal_ngram3_alpha4.csv``,
``word_surprisals_gpt2.csv``, ``word_surprisals_yugo.csv``,
``word_surprisals_bert.csv``, ``word_surprisals_bertic.csv``)
plus the unigram log-probability table
``../podaci/wordlist_frequencies.csv``, merges them on
``Sentence`` / ``Word``, min-max normalises every numeric column,
and for each unique target sentence draws a line plot showing the
five surprisal estimators side-by-side along the words of the
sentence (x = word, y = normalised surprisal). Used to qualitatively
compare model behaviour at the sentence level.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


# Read surprisal data
data_paths = []
data_paths.append(os.path.join('..','podaci', 'surprisal values', 'word_surprisal_ngram3_alpha4.csv'))
data_paths.append(os.path.join('..','podaci', 'surprisal values', 'word_surprisals_gpt2.csv'))
data_paths.append(os.path.join('..','podaci', 'surprisal values', 'word_surprisals_yugo.csv'))
data_paths.append(os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bert.csv'))
data_paths.append(os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bertic.csv'))

surprisal_list = ['Surprisal ngram-3',
                  'Surprisal GPT-2',
                  'Surprisal Yugo',
                  'Surprisal BERT',
                  'Surprisal BERTic'  
    ]

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
    """Min-max normalise a numeric ``pandas`` column to [0, 1].

    Computes ``(column - column.min()) / (column.max() - column.min())``
    elementwise. ``NaN`` values are propagated by ``pandas`` arithmetic
    (any ``NaN`` minus a finite minimum stays ``NaN``). If
    ``column.max() == column.min()`` the result is all-``NaN`` (division
    by zero); the caller is responsible for filtering such columns.

    Parameters
    ----------
    column : pandas.Series
        Numeric column to rescale.

    Returns
    -------
    pandas.Series
        Same index as ``column`` with values rescaled into ``[0, 1]``
        based on the column-wide minimum and maximum.
    """
    return (column - column.min()) / (column.max() - column.min())

columns = data.columns
for column in columns[3:]:
    data[column] = normalize_column(data[column])
    
# Plot each sentence on separate plots
for sentence_id in data['Sentence'].unique():
    
    try:
        plt.figure(figsize=(12,6))
        
        group = data[data['Sentence'] == sentence_id]
        sentence = target_sentences_df['Text'][sentence_id].split()
        
        for surprisal_name in surprisal_list:
            
            surprisal_values = []
            for word in sentence:
                surprisal_value = group[group['Word']==word][surprisal_name].values[0]
                surprisal_values.append(surprisal_value)
                
            plt.plot(sentence, surprisal_values, linewidth = 3)
            plt.scatter(sentence, surprisal_values, label=surprisal_name[10:], marker='o', s=80)
       
        plt.xlabel('word', fontsize=20)
        plt.ylabel('surprisal', fontsize=20)
        plt.xticks(rotation=45, fontsize=20)  # Set xticks fontsize and rotation
        plt.legend(fontsize=20, bbox_to_anchor = (1.3, 0.6), loc='center right')
        plt.yticks(fontsize=20)
        plt.title('Surprisal Values in Example Sentence', fontsize= 25)
        plt.tight_layout()
        
        plt.show()
        
    except:
        print(f"Error while processing sentence: {sentence_id}")


