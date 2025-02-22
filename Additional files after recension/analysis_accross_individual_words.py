# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:06:43 2025

@author: Jelena
"""

from my_functions import inf_k_model, akaike_for_column
import numpy as np
import pandas as pd
import os

file_path =  os.path.join('..','podaci', 'training_data.csv')
df = pd.read_csv(file_path)

ts_file_path =  os.path.join('..','podaci', 'target_sentences.csv')
ts_df = pd.read_csv(ts_file_path)

pt_file_path = os.path.join('..','podaci', 'pos_tags.csv')
pt_df = pd.read_csv(pt_file_path)

columns_to_remove = ['surprisal ngram2 alpha4', 'surprisal ngram3 alpha4', 'surprisal ngram4 alpha4',
                     'surprisal ngram5 alpha4', 'surprisal ngram2 alpha20', 'surprisal ngram3 alpha20', 
                     'surprisal ngram4 alpha20', 'surprisal ngram5 alpha20','surprisal BERT', 
                     'surprisal BERTic', 'surprisal GPT3', 'surprisal yugo']

df = df.drop(columns = columns_to_remove)


# Assuming ts_df has columns: 'target sentence' and 'Text'
df["word count"] = df["target sentence"].apply(lambda idx: len(str(ts_df.iloc[idx]["Text"]).split()) if 0 <= idx < len(ts_df) else 0)

''' Add pos tag '''

pos_tag_list = []
for _, row in df.iterrows():
    
    words = row['word'].lower().strip()
    
    word_pos_tag = ''
    for word in words.split(' '):
        pos_tag = pt_df[pt_df['word'] == word]['pos tag'].iloc[0]
        word_pos_tag += ' ' + pos_tag
        
    pos_tag_list.append(word_pos_tag.strip())
    
df['pos tag'] = pos_tag_list

''' Calulate delta LL '''

import warnings
# Filter out SettingWithCopyWarning
warnings.filterwarnings("ignore")
x_axis = np.arange(0.25, 3, 0.25)

for i in x_axis:
  k = round(i, 2)
  df = inf_k_model(df, k, 'surprisal GPT')

# Reset warnings to default behavior (optional)
warnings.resetwarnings() 

column_name_1 = 'pos tag'
column_name_2 = 'emotion'

ll_best_improvement = []
the_best_k = []
ll_k1_improvement = []
column_1_list = []
column_2_list = []

filtered_list = [item for item in df[column_name_1].unique().tolist() if len(item.split(" ")) <= 1]

#for  column_1 in df[column_name_1].unique().tolist():
#for  column_1 in ['content', 'function', 'content function' or 'function content']:
for column_1 in filtered_list:
    
    for column_2 in df[column_name_2].unique().tolist():
    
        column_1_list.append(column_1)
        column_2_list.append(column_2)
        
        filtered_data = df[(df[column_name_1] == column_1) & (df[column_name_2] == column_2)]
        
        k_list = []
        k_improvements = []
        for i in x_axis:
            k = round(i, 2)
            k_list.append(k)
            
            difference = akaike_for_column(filtered_data,  f"surprisal GPT {str(k)} model", 'baseline')
            k_improvements.append(difference)
            
        max_value = max(k_improvements)  # Find the max value
        max_index = k_improvements.index(max_value)  # Find the index of the max value
    
        ll_best_improvement.append(max_value)
        the_best_k.append(k_list[max_index])
        ll_k1_improvement.append(k_improvements[3])


# Create DataFrame
df_grouped = pd.DataFrame({
    "LL": ll_k1_improvement,
    "optimal k": the_best_k,
    "LL the best": ll_best_improvement,
    column_name_1: column_1_list,
    column_name_2: column_2_list
})

''' Anlyze results '''

import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
df = df_grouped

sns.scatterplot(x=df[column_name_1], y=df['LL'], hue=df[column_name_2], palette='viridis')
#plt.xticks(rotation=45, ha='right')
plt.xlabel('pos tag')
plt.ylabel('LL')
plt.show()

