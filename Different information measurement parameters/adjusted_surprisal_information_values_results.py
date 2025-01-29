# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:18:41 2024

@author: Jelena
"""


from my_functions import add_column_with_surprisal, paired_permutation_test, calculate_delta_ll
import pandas as pd
import os


file_path = os.path.join('..','podaci','information measurements parameters', "data.csv") 
data = pd.read_csv(file_path)
data = data[data['time']!=0]
    
results_list = []
parameter_list = []
emotion_list = []
std_list = []

surprisal = 'Surprisal GPT-2'
#surprisal = 'None'

#parameters = ['Contextual Entropy']
              
parameters = ['Context Embedding','Non-context Embedding','Pos-Tag','Orthographic']

#parameters = ['AS Context Embedding','AS Non-context Embedding','AS Pos-Tag','AS Orthographic']

# Add surprisal results
df = data.dropna()

if parameters[0] == 'Contextual Entropy':

    results_df = add_column_with_surprisal(data, surprisal, None, 3)
    data = pd.merge(data, results_df, how='left')

    results_df = add_column_with_surprisal(data, 'Contextual Entropy', None, 3)
    data = pd.merge(data, results_df, how='left')

    for emotion in [0,1,2,3,4]:
        emotion_data = df[df['emotion'] == emotion]
                
        delta_element, _ = calculate_delta_ll(emotion_data,  f"None + {surprisal} model")
        std_element = paired_permutation_test(emotion_data, 'baseline -3', f"None + {surprisal} model", 100)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append(surprisal)
    
        delta_element, _ = calculate_delta_ll(emotion_data,  "None + Contextual Entropy model")
        std_element = paired_permutation_test(emotion_data, 'baseline -3', "None + Contextual Entropy model", 100)
        results_list.append(delta_element)
        std_list.append(std_element)
        emotion_list.append(emotion)
        parameter_list.append("Contextual Entropy model")
            
    
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

''' Print results '''
    
parameters = set(results_df['parameter'])

x_axis = [0, 1, 2, 3, 4]

for p in parameters:
    print(p)
    for emotion in x_axis:
            emotion_data = results_df[results_df['emotion'] == emotion]
            value = emotion_data[emotion_data['parameter']==p]['y_axis'].values[0]
            p_value = emotion_data[emotion_data['parameter']==p]['p_value'].values[0]
            print(f"{value:.3f} {p_value:.3f}")

