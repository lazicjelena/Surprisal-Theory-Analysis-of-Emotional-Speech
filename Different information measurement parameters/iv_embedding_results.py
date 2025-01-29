# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:15:20 2024

@author: Jelena
"""

from my_functions import add_column_with_surprisal, paired_permutation_test, calculate_delta_ll
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import warnings


context_ind = True # otherwise non-context
surprisal_ind = False # otherwise information value
serbian_ind = False # otherwise english

if context_ind:
    parameter_name = 'CE'
    if surprisal_ind:
        file_path = os.path.join('..','podaci','information measurements parameters', "context_embedding_data_surprisal.csv")
    else:
        file_path = os.path.join('..','podaci','information measurements parameters', "context_embedding_data.csv")   
else:
    parameter_name = 'NCE'
    if surprisal_ind:
        file_path = os.path.join('..','podaci','information measurements parameters', "non_context_embedding_data_surprisal.csv")
    else:
        file_path = os.path.join('..','podaci','information measurements parameters', "non_context_embedding_data.csv")

if serbian_ind:
    emotion_names = ['неутрално', 'срећно', 'тужно', 'уплашено', 'љуто']
    x_axis_name = 'дубина неуралне мреже'
    p_names = ['p вриједност<0.001', 'p бриједност<0.05']
else:
    emotion_names = ['neutral', 'happy', 'sad', 'scared', 'angry']
    x_axis_name = 'network layer'
    p_names = ['p value<0.001', 'p value<0.05']
    
emotion_colours = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}

# Ignore all warnings
warnings.filterwarnings("ignore")

parameters = [f'{parameter_name} {j}' for j in range(1, 13)]

data = pd.read_csv(file_path)
data = data[data['time']!=0]
    
results_list = []
parameter_list = []
emotion_list = []
std_list = []

surprisal = 'Surprisal GPT-2'

# Add surprisal results
df = data.dropna()

for parameter in parameters:
    
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
            
    for emotion in [0,1,2,3,4]:
        emotion_data = df[df['emotion'] == emotion]

        delta_element, _ = calculate_delta_ll(emotion_data,  f"{surprisal} + {parameter} model")
        std_element = paired_permutation_test(emotion_data, 'baseline -3', f"{surprisal} + {parameter} model", 100)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append(surprisal + ' ' + parameter)
            
results_df = pd.DataFrame({'y_axis': results_list,
                          'p_value': std_list,
                          'parameter': parameter_list,
                          'emotion': emotion_list})

''' plot data '''
# Define legend handles for each emotion
legend_elements = []
for i in [0, 1, 2, 3 ,4]:
    legend_elements.append(Line2D([0], [0], color=emotion_colours[i], lw=2, label=emotion_names[i]))
    
legend_elements.append(Line2D([0], [0], color='grey', marker='*', lw=0, markersize=20, label= p_names[0]))
legend_elements.append(Line2D([0], [0], color='grey', marker='.', lw=0, markersize=20, label= p_names[1]))
    

fig = plt.figure(figsize=(12,6))

for emotion in [0,1,2,3,4]:
    emotion_df = results_df[results_df['emotion'] == emotion]
    
    y_axis = []
    x_axis = []
    
    scatter_y_axis = []
    scatter_x_axis = []
    
    large_scatter_y_axis = []
    large_scatter_x_axis = []
    
    for i in range(1,13):
        value = emotion_df.loc[emotion_df['parameter'] == f"Surprisal GPT-2 {parameter_name} {i}", 'y_axis'].values[0] 
        p_value = emotion_df.loc[emotion_df['parameter'] == f"Surprisal GPT-2 {parameter_name} {i}", 'p_value'].values[0]
        
        x_axis.append(i)
        y_axis.append(value)
        
        if p_value<0.001:
            large_scatter_y_axis.append(value)
            large_scatter_x_axis.append(i)
        else:
            if p_value<0.05:
                scatter_y_axis.append(value)
                scatter_x_axis.append(i)
        
    
    plt.plot(x_axis, y_axis, linewidth=3, color = emotion_colours[emotion])  
    plt.scatter(large_scatter_x_axis, large_scatter_y_axis, marker='*', s=250, color = emotion_colours[emotion])
    plt.scatter(scatter_x_axis, scatter_y_axis, s=80, color = emotion_colours[emotion])
    plt.tick_params(axis='both', which='major', labelsize=15)


fig.text(0.4, 0.001, x_axis_name, ha='center', va='center', fontsize=25)
fig.text(0.0001, 0.5, r'$\Delta$LogLikelihood', ha='center', va='center', rotation='vertical', fontsize=25)
plt.legend(handles=legend_elements, fontsize=15, loc="center left", bbox_to_anchor=(1, 0.5))


plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


''' print last hidden layer'''

for emotion in [0,1,2,3,4]:
    
    emotion_df = results_df[results_df['emotion'] == emotion]
    print(emotion)
    i = 12
    
    value = emotion_df.loc[emotion_df['parameter'] == f"Surprisal GPT-2 {parameter_name} {i}", 'y_axis'].values[0] 
    p_value = emotion_df.loc[emotion_df['parameter'] == f"Surprisal GPT-2 {parameter_name} {i}", 'p_value'].values[0]
        
    print(f"{value:.3f} {p_value}")

