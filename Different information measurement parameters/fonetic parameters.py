# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:26:07 2024

@author: Jelena
"""

from my_functions import add_column, fonetic_model, paired_permutation_test, calculate_delta_ll
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
import os

df_path = os.path.join('..','podaci', 'training data', "Surprisal GPT-2.csv") 
data = pd.read_csv(df_path)
data = data.drop(columns=['speaker'])
                         
# Add new column for fonetci features
ff_path = os.path.join('..','podaci', 'information measurements parameters', "fonetic_features1.csv") 
ff_data = pd.read_csv(ff_path)
ff_data = ff_data.drop('sentence', axis=1)
ff_data = ff_data.drop_duplicates()

data = data.merge(ff_data, how='left')

columns_list = ['Surprisal GPT-2',
                'length',
                'log probability'
                ]

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

''' Make baseline model '''
# Ignore all warnings
warnings.filterwarnings("ignore")

data = data.replace("nan", np.nan)
data = data[data['time']!=0]

results_df = add_column(data, 3)    
data = pd.merge(data, results_df, how='left')

''' Add fonetic parameters '''

results_list = []
parameter_list = []
emotion_list = []
std_list = []

#fonem_types = ['vokali','sonanti', 'konsonanti']
               
#fonem_types = ['vokali', 'alveolarni', 'palatalni', 'bilabijalni',
#               'labio_dentalni', 'labialni', 'zubni', 'palatalni', 'zadnjonepcani']

#fonem_types = ['zvucni', 'bezvucni']

fonem_types = ['vokali', 'alveolarni', 'palatalni', 'bilabijalni', 'labio_dentalni',
               'labialni', 'zubni', 'palatalni', 'zadnjonepcani', 'zvucni', 'bezvucni']


pom_df = fonetic_model(data, fonem_types)
data = pd.merge(data, pom_df, how='left')
df = data.dropna()
    
for emotion in [0,1,2,3,4]:
    emotion_data = df[df['emotion'] == emotion]
            
    delta_element, _ = calculate_delta_ll(emotion_data, "fonetic model")
    std_element = paired_permutation_test(emotion_data, 'baseline -3', 'fonetic model')
    results_list.append(delta_element)
    std_list.append(std_element)
    emotion_list.append(emotion)
    parameter_list.append('Fonetic')
            
            
results_df = pd.DataFrame({'y_axis': results_list,
                           'p_value': std_list,
                           'parameter': parameter_list,
                           'emotion': emotion_list})

''' Print results '''

parameters = ['Fonetic']
x_axis = [0, 1, 2, 3, 4]


for p in parameters:
    for emotion in x_axis:
            emotion_data = results_df[results_df['emotion'] == emotion]
            value = emotion_data[emotion_data['parameter']==p]['y_axis'].values[0]
            p_value = emotion_data[emotion_data['parameter']==p]['p_value'].values[0]
            print(f"{value:.3f} {p_value:.3f}")
      
        
''' Plot feature distribution '''
# Assuming df is your dataframe with the specified columns
# Summing the values of each column
column_sums = df[fonem_types].sum()

# Plotting the results in a bar chart
plt.figure(figsize=(10, 6))
column_sums.plot(kind='bar', color='blue')
plt.title('Sum of Each Phonetic Feature', fontsize = 25)
plt.xlabel('Phonetic Feature', fontsize = 20)
plt.ylabel('Sum', fontsize = 20)
plt.xticks(rotation=45, fontsize = 15)
plt.show()

