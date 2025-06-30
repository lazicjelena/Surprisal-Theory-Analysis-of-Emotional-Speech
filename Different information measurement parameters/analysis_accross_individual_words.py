# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:02:06 2025

@author: Jelena
"""
 
from my_functions import add_column_with_surprisal, calculate_delta_ll
import pandas as pd
import os

''' Read data '''

model = 'bertic'
surprisal = 'surprisal BERTic uni'

# model = 'bert'
# surprisal = 'surprisal BERT uni'

# model = 'gpt'
# surprisal = 'surprisal GPT'

baseline_model = 'baseline -3'

file_path =  os.path.join('..','podaci','transformer layers parameters', 'datasets', f"{model}.csv")
df = pd.read_csv(file_path)

ts_file_path =  os.path.join('..','podaci', 'target_sentences.csv')
ts_df = pd.read_csv(ts_file_path)

pt_file_path = os.path.join('..','podaci', 'pos_tags.csv')
pt_df = pd.read_csv(pt_file_path)

''' Add baseline results column'''


column_list = ['word', 'speaker', 'emotion', 'time', 'position', 'target sentence',
               'speaker gender', 'length', 'log probability', 
               surprisal,  'word type', 'fold', 'baseline']

#df = df[column_list]


# Assuming ts_df has columns: 'target sentence' and 'Text'
df["Sentence Word Count"] = df["target sentence"].apply(lambda idx: len(str(ts_df.iloc[idx]["Text"]).split()) if 0 <= idx < len(ts_df) else 0)

''' Add pos tag '''

df = df[df['word type'].astype(str).isin(['content', 'function'])]

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
filtered_list = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PUNCT', 'FUNC']

parameters = [f'CE {j}' for j in range(1, 13)]

results_list = []
parameter_list = []
postag_list = []
 
for parameter in parameters:
    
    # estimate spoken time duration time using all parameters including models surprisal
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
    
    for postag in filtered_list:
            emotion_data = df[df['PoS tag'] == postag]

            delta_element, _ = calculate_delta_ll(emotion_data,  f"{surprisal} {parameter} model")
            
            results_list.append(delta_element)
            postag_list.append(postag)
            parameter_list.append(surprisal + ' ' + parameter)
            
results_df = pd.DataFrame({'y_axis': results_list,
                          'parameter': parameter_list,
                          'postag': postag_list})

''' Plot Results '''
import matplotlib.pyplot as plt
 
word_colors = {
    'NOUN': (1, 0, 0, 1),       # red
    'VERB': (0, 0, 1, 1),       # blue
    'ADJ': (0, 1, 0, 1),       # green
    'ADV': (1, 0.65, 0, 1),    # orange (RGB ≈ 255,165,0)
    'PUNCT': (0.5, 0, 0.5, 1),    # purple (RGB ≈ 128,0,128)
    'FUNC': (0, 0.65, 1, 1), 
}

# Ako je već učitan, samo modifikujemo kolonu "parameter"
results_df['parameter'] = results_df['parameter'].str.extract(r'(\d+)$').astype(int)

# Plotovanje
plt.figure(figsize=(10, 6))

for postag in filtered_list:
    subset = results_df[results_df['postag'] == postag]
    plt.plot(subset['parameter'], subset['y_axis'], marker='o', linewidth = 3, label=postag, color=word_colors[postag])

plt.xlabel("network layer", fontsize = 20)
plt.ylabel("ΔLogLikelihood", fontsize = 20)
plt.legend(fontsize = 15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()

        