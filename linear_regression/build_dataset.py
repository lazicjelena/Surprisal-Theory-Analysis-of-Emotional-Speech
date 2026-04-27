# -*- coding: utf-8 -*-
"""build_dataset.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Aggregates per-speaker word-level feature CSVs together with the
per-word surprisal CSVs produced by every upstream surprisal
estimator (GPT-2, YugoGPT, GPT-3 / GPT-Neo, BERT, BERTic, and the
n-gram models with ``alpha=4`` and ``alpha=20`` for
``n in {2, 3, 4, 5}``) into a single master training table
``../podaci/training_data.csv``. Per-word features and per-word
surprisals are joined sentence-by-sentence with
:func:`lookup_features` (numeric columns) and
:func:`add_word_type` (string columns) and the rows are
fold-partitioned. The resulting CSV is consumed downstream by
``baseline_model.py`` (which adds the ``baseline`` column),
``residual_distribution.py``, ``final_graphs.py``, and
``results.py``.

"""

import os
import pandas as pd

def lookup_features(data, freq_df, column_name):
    """Per-word numeric feature lookup from a per-sentence frequency table.

    Iterates over the per-spoken-word rows of ``data`` and, for the
    word(s) encoded in the ``word`` column at the given
    ``target sentence``, looks up the corresponding numeric value
    from ``freq_df[column_name]`` (matched by ``Sentence`` ==
    ``target sentence`` and ``Word`` == ``word``). When a row of
    ``data`` carries several space-separated words (the lemma /
    surface variant case), the looked-up values are summed and the
    sum is returned for that row. Per-sentence repeated words are
    disambiguated by occurrence index.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-spoken-word table; must contain ``word`` and
        ``target sentence``.
    freq_df : pandas.DataFrame
        Per-(sentence, word) feature table from one upstream
        surprisal CSV. Must contain ``Sentence``, ``Word``, and
        ``column_name`` columns.
    column_name : str
        Name of the numeric feature column to lift across.

    Returns
    -------
    list of float
        ``len(data)`` entries; the per-row sum over the matched
        ``column_name`` values, ``0`` when no row of ``freq_df``
        matches.
    """
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

def add_word_type(data, freq_df, column_name):
    """Per-word categorical (string) feature lookup.

    String-valued counterpart of :func:`lookup_features`. For every
    row of ``data`` and every space-separated word in
    ``row['word']``, looks up the matching ``Sentence``/``Word`` row
    in ``freq_df`` and concatenates the ``column_name`` value
    (separated by single spaces) into a per-row string. Trailing /
    leading whitespace is stripped before append.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-spoken-word table; same shape as in
        :func:`lookup_features`.
    freq_df : pandas.DataFrame
        Per-(sentence, word) feature table; ``column_name`` is a
        string-valued column (e.g. word type).
    column_name : str
        Name of the categorical feature column to lift across.

    Returns
    -------
    list of str
        ``len(data)`` entries; per-row whitespace-joined string.
    """
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
        log_probability_value = ''
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq_s = freq_df[freq_df['Sentence'] == sentence]
            freq = freq_s[freq_s['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            if not freq.empty:
                log_probability_value += ' '
                log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
            else:
              log_probability_value += ' '
              print('error')
              print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s):
              list_of_words = []

        log_prob_list.append(log_probability_value.strip())

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

#log_probab_list = lookup_features(data, freq_df, 'Log Probability')
#data['log probability'] = log_probab_list

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