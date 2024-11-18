# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

"""

import pandas as pd
import os

df_path = os.path.join('..','podaci', 'training data', "Surprisal GPT-2.csv") 
data = pd.read_csv(df_path)
data = data.drop(columns=['speaker'])
                         
# Add new column for contextual entropy
ce_df_path = os.path.join('..','podaci', 'information measurements parameters', "contextual_entropy.csv") 
ce_data = pd.read_csv(ce_df_path)

# Add new column for information_values
iv_df_path = os.path.join('..','podaci', 'information measurements parameters', "information_value.csv") 
iv_data = pd.read_csv(iv_df_path)

# Add new column for adjusted_surprisals
as_df_path = os.path.join('..','podaci', 'information measurements parameters', "adjusted_surprisal.csv") 
as_data = pd.read_csv(as_df_path)

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


contextual_entropy_list = lookup_features(data, ce_data, 'Contextual Entropy')
data['Contextual Entropy'] = contextual_entropy_list

information_values_list = ['Context Embedding',
                           'Non-context Embedding',
                           'Pos-Tag',
                           'Orthographic'
                           ]


adjusted_surprisals_list = ['AS Context Embedding',
                           'AS Non-context Embedding',
                           'AS Pos-Tag',
                           'AS Orthographic'
                           ]

for iv in information_values_list:
    iv_list = lookup_features(data, iv_data, iv)
    data[iv] = iv_list
    
for AS in adjusted_surprisals_list:
    as_list = lookup_features(data, as_data, AS)
    data[AS] = as_list

columns_list = ['Contextual Entropy',
                'Surprisal GPT-2',
                'length',
                'log probability'
                ]

columns_list = columns_list + information_values_list + adjusted_surprisals_list

for column in columns_list:
    # Add new columns for the length and log probability of the previous word
    data[f"{column} -1"] = data[f"{column}"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -1"]] = pd.NA
    data[f"{column} -2"] = data[f"{column} -1"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -2"]] = pd.NA
    data[f"{column} -3"] = data[f"{column} -2"].shift(1)
    data.loc[data['target sentence'] != data['target sentence'].shift(1), [f"{column} -3"]] = pd.NA
    

column_names = data.columns.tolist()
column_names = [x for x in column_names if x != 'time'] 
# Group by all columns except 'time' and calculate the mean for 'time'
#data = data.groupby(column_names, as_index=False).agg({'time': 'mean'})
                         
# Save the concatenated data to a CSV file
output_csv_path = os.path.join('..','podaci','information measurements parameters', "data.csv") 
data.to_csv(output_csv_path, index=False)

