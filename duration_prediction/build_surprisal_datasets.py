# -*- coding: utf-8 -*-
"""build_surprisal_datasets.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Per-model surprisal column joiner for the duration-regression
analysis. Reads the master training table
``../podaci/training data/general_data.csv`` (one row per spoken
word produced by ``transform_data_into_dataframe.py``) and, for
each surprisal source CSV in ``../podaci/surprisal values/``
(GPT-2, Yugo, BERT, BERTic, ngram-3), uses
:func:`lookup_features` to build a per-row surprisal vector keyed
by ``(target sentence, word)`` with positional handling for
repeated words inside the same sentence. Each variant is written
to ``../podaci/training data/<Surprisal X>.csv`` so that
``surprisal_results.py`` can train one regression per surprisal.

"""

import os
import pandas as pd

def lookup_features(data, freq_df, column_name):
    """Build a per-row summed surprisal value keyed by ``(target sentence, word)``.

    For every row of ``data`` the words in ``data['word']``
    (a single spoken token, possibly hyphenated as multiple
    orthographic words separated by spaces) are looked up in
    ``freq_df`` filtered by the matching ``Sentence``. Repeated
    occurrences of the same word inside a sentence are resolved
    positionally by counting how many times the same word has
    already been consumed in the current sentence, so the
    ``i``-th occurrence picks the ``i``-th matching row from
    ``freq_df``. The returned list contains the summed
    ``column_name`` value across all space-separated sub-words of
    a row, with ``0`` substituted (and an ``error`` print) when a
    word is missing from ``freq_df``.

    Parameters
    ----------
    data : pandas.DataFrame
        Master per-word table with at least ``word`` and
        ``target sentence`` columns.
    freq_df : pandas.DataFrame
        Per-(sentence, word) lookup table with at least
        ``Sentence``, ``Word`` and ``column_name`` columns.
    column_name : str
        Name of the surprisal/log-probability column in
        ``freq_df`` to extract.

    Returns
    -------
    list of float
        One summed surprisal value per row of ``data``, in input
        order.
    """
    log_prob_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        
        words = row['word'].split(' ')
        sentence = row['target sentence']
        log_probability_value = 0
        
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
        
        
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
base_path = os.path.join('..','podaci','training data', 'general_data.csv') 
data = pd.read_csv(base_path)


surprisal_data_name = ['word_surprisals_gpt2.csv', 
                       'word_surprisals_yugo.csv',
                       'word_surprisals_bert.csv',
                       'word_surprisals_bertic.csv',
                       'word_surprisal_ngram3_alpha4.csv'
                       ]

surprisal_column_name = ['Surprisal GPT-2',
                         'Surprisal Yugo', 
                         'Surprisal BERT',
                         'Surprisal BERTic',
                         'Surprisal ngram-3'
                         ]

for df_name, column_name in zip(surprisal_data_name, surprisal_column_name):
    
    df_path = os.path.join('..','podaci', 'surprisal values', df_name) 
    surprisal_df = pd.read_csv(df_path)
    
    surprisal_values_list = lookup_features(data, surprisal_df, column_name)
    data[column_name] = surprisal_values_list
    
    # Save the concatenated data to a CSV file
    output_csv_path = os.path.join('..','podaci', 'training data', column_name + '.csv') 
    data.to_csv(output_csv_path, index=False)
    data.drop(columns=[column_name], inplace=True)
    
    
    