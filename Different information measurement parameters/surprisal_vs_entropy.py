# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:05:54 2024

@author: Jelena
"""

from my_functions import add_column_with_surprisal, calculate_delta_ll
import pandas as pd
import os

file_path = os.path.join('..','podaci','information measurements parameters', "data.csv") 
df = pd.read_csv(file_path)
df = df[df['time']!=0]
    
results_list = []
parameter_list = []
emotion_list = []
gender_list = []
std_list = []

surprisal = 'Surprisal GPT-2'

parameters = ['Contextual Entropy',
              'Context Embedding',
              'Non-context Embedding',
              'Pos-Tag',
              'Orthographic',
              'AS Context Embedding',
              'AS Non-context Embedding',
              'AS Pos-Tag',
              'AS Orthographic'
                ]

# Add surprisal results

results_df = add_column_with_surprisal(df, surprisal, None, 3)
df = pd.merge(df, results_df, how='left')
    
for gender in ['f', 'm']:
    gender_data = df[df['speaker gender'] == gender]
            
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion'] == emotion]
            
        delta_element, std_element = calculate_delta_ll(emotion_data,  f"None + {surprisal} model")
        gender_list.append(gender)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append(surprisal)
            
for parameter in parameters:
    
    results_df = add_column_with_surprisal(df, parameter, None, 3)
    df = pd.merge(df, results_df, how='left')
    
    results_df = add_column_with_surprisal(df, parameter, surprisal, 3)
    df = pd.merge(df, results_df, how='left')
    
    for gender in ['f', 'm']:
        gender_data = df[df['speaker gender'] == gender]
            
        for emotion in [0,1,2,3,4]:
            emotion_data = gender_data[gender_data['emotion'] == emotion]
            
            delta_element, std_element = calculate_delta_ll(emotion_data,  f"None + {parameter} model")
            gender_list.append(gender)
            results_list.append(delta_element)
            std_list.append(std_element)
            emotion_list.append(emotion)
            parameter_list.append(parameter)
            
            delta_element, std_element = calculate_delta_ll(emotion_data,  f"{surprisal} + {parameter} model")
            gender_list.append(gender)
            results_list.append(delta_element)
            std_list.append(std_element)
            emotion_list.append(emotion)
            parameter_list.append(surprisal + ' ' + parameter)
            
data = {'y_axis': results_list,
        'y_std': std_list,
        'parameter': parameter_list,
        'emotion': emotion_list,
        'speaker gender': gender_list}
    
results_df = pd.DataFrame(data)
results_path = os.path.join('..','podaci','information measurements parameters', "results.csv") 
results_df.to_csv(results_path, index=False)


