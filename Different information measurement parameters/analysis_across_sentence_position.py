# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:10:31 2025

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
df = df.dropna()
df = df[df['time']!=0]

ts_file_path =  os.path.join('..','podaci', 'target_sentences.csv')
ts_df = pd.read_csv(ts_file_path)

pt_file_path = os.path.join('..','podaci', 'pos_tags.csv')
pt_df = pd.read_csv(pt_file_path)


''' Calculate LL improvement '''

parameters = [f'CE {j}' for j in range(1, 13)]

results_list = []
parameter_list = []
position_list = []
 
for parameter in parameters:
    
    # estimate spoken time duration time using all parameters including models surprisal
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
    
    for position in df['position'].unique().tolist():
        
            emotion_data = df[df['position'] == position]

            delta_element, _ = calculate_delta_ll(emotion_data,  f"{surprisal} {parameter} model")
            
            results_list.append(delta_element)
            position_list.append(position)
            parameter_list.append(surprisal + ' ' + parameter)
            
results_df = pd.DataFrame({'y_axis': results_list,
                          'parameter': parameter_list,
                          'position': position_list})

''' Plot Results '''
import matplotlib.pyplot as plt

# Tvoj DataFrame (pretpostavljam da se zove results_df)
# Ako je već učitan, samo modifikujemo kolonu "parameter"
results_df['parameter'] = results_df['parameter'].str.extract(r'(\d+)$').astype(int)

# Plotovanje
plt.figure(figsize=(10, 6))
positions = ['b', 'm', 'e']
colors = {'b': 'blue', 'm': 'green', 'e': 'red'}
labels = {'b': 'beginning', 'm': 'middle', 'e': 'end'}

for pos in positions:
    subset = results_df[results_df['position'] == pos]
    plt.plot(subset['parameter'], subset['y_axis'], marker='o', linewidth = 3, label=labels[pos], color=colors[pos])

plt.xlabel("network layer", fontsize = 20)
plt.ylabel("ΔLogLikelihood", fontsize = 20)
plt.legend(fontsize = 15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()

    