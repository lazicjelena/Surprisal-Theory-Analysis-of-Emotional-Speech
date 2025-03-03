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

column_list = ['word', 'speaker', 'emotion', 'time', 'position', 'target sentence',
               'speaker gender', 'length', 'log probability', 
               'surprisal GPT',  'word type', 'fold', 'baseline']
     
df = df[column_list]
df = df.rename(columns={'position': 'Word Position in Sentence'})

ts_file_path =  os.path.join('..','podaci', 'target_sentences.csv')
ts_df = pd.read_csv(ts_file_path)

pt_file_path = os.path.join('..','podaci', 'pos_tags.csv')
pt_df = pd.read_csv(pt_file_path)

# Assuming ts_df has columns: 'target sentence' and 'Text'
df["Sentence Word Count"] = df["target sentence"].apply(lambda idx: len(str(ts_df.iloc[idx]["Text"]).split()) if 0 <= idx < len(ts_df) else 0)

''' Add pos tag '''

df = df[df['word type'].astype(str).isin(['content', 'function'])]

# pos_tag_list = []
# for _, row in df.iterrows():
    
#     words = row['word'].lower().strip()
    
#     word_pos_tag = ''
#     for word in words.split(' '):
#         pos_tag = pt_df[pt_df['word'] == word]['pos tag'].iloc[0]
#         word_pos_tag += ' ' + pos_tag
        
#     pos_tag_list.append(word_pos_tag.strip())


pos_tag_list = []
for _, row in df.iterrows():
    
    word = row['word'].lower().strip()
    
    if row['word type'] == 'function':
        pos_tag = 'FUNC'
    else:
        pos_tag = pt_df[pt_df['word'] == word]['pos tag'].iloc[0]
        
    pos_tag_list.append(pos_tag)
    
df['PoS tag'] = pos_tag_list

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

column_name_1 = 'PoS tag'
filtered_list = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PUNCT', 'FUNC']

column_name_1 = "Sentence Word Count"
filtered_list = df[column_name_1].unique().tolist()
filtered_list.remove(2)

column_name_1 = 'Word Position in Sentence'
filtered_list = df[column_name_1].unique().tolist()

column_name_2 = 'emotion'
#column_name_2 = "Sentence Word Count"

ll_best_improvement = []
the_best_k = []
ll_k1_improvement = []
column_1_list = []
column_2_list = []


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
            
            difference, _ = akaike_for_column(filtered_data,  'time', f"surprisal GPT {str(k)} model", 'baseline')
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


plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_grouped[column_name_1], y=df_grouped['LL the best'], hue=df_grouped[column_name_2], palette='viridis', s=100)
# Poboljšanja u čitljivosti
plt.xlabel(column_name_1, fontsize=20)
plt.ylabel(r'$\Delta LL_{k=1}$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Poboljšana legenda
handles, labels = plt.gca().get_legend_handles_labels()
labels = ['neutral', 'happy', 'sad', 'scared', 'angry']
plt.legend(handles=handles, labels=labels, 
           title='Emotion', fontsize=15, title_fontsize=20, loc='best')
plt.show()
