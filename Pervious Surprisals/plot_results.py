# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 08:47:22 2024

@author: Jelena

Pipeline role
-------------
Final plotting stage of the "previous surprisals" analysis. Reads
the merged prominence + lagged-surprisal table
``../podaci/correlation data/full_former_surprisal_data.csv``
(produced by ``conjoint_data.py``), drops rows with missing
prosodic parameter (``time``, ``energy`` or ``f0`` -- selected by
the module-level ``prominence_parameter``), and for every surprisal
model and every ``(gender, emotion)`` cell computes the multiple
correlation coefficient between ``[<Model>, <Model> k=1, ...,
<Model> k=K]`` and the prosodic parameter for ``K = 0..10``.
Renders four 2x5 subplot grids: GPT-2 / Yugo / ngram-3 in English,
BERT / BERTic in English, and the same two plots in Cyrillic
Serbian. Used to produce the figures that go into the chapter on
the "previous surprisal" effect.
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import warnings
import os

prominence_parameter = 'time'
data_csv_path = os.path.join('..','podaci', 'correlation data', "full_former_surprisal_data.csv") 
prominence_df = pd.read_csv(data_csv_path)

def calculate_corr_coef(x,y):
    """Compute the multiple correlation coefficient via OLS R-squared.

    Fits :class:`sklearn.linear_model.LinearRegression` on ``x`` vs.
    ``y``, computes the predicted ``y_pred``, and returns the square
    root of :func:`sklearn.metrics.r2_score(y, y_pred)`. Handles two
    input shapes: a 1-D ``x`` is reshaped to ``(-1, 1)`` and treated
    as a single regressor, while a 2-D ``x`` (DataFrame or array)
    is used as-is and treated as multiple regressors -- in which
    case the returned value is the multiple correlation coefficient
    between the regressors and ``y``. Pandas / sklearn warnings are
    silenced for the duration of the call.

    Parameters
    ----------
    x : numpy.ndarray or pandas.DataFrame
        Predictor(s). 1-D or 2-D.
    y : numpy.ndarray
        Target values, 1-D.

    Returns
    -------
    float
        ``sqrt(R^2)`` of the OLS fit.

    Side effects
    ------------
    Calls ``warnings.filterwarnings("ignore")`` at the start and
    ``warnings.resetwarnings()`` before returning to silence
    sklearn / pandas chained-assignment chatter.
    """
    warnings.filterwarnings("ignore")
    
    if len(x.shape)==1:
        # Create and fit the regression model
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        # Predict Y values based on the model
        y_pred = model.predict(x.reshape(-1, 1))    
    else:
        # Create and fit the regression model
        model = LinearRegression()
        model.fit(x, y)
        # Predict Y values based on the model
        y_pred = model.predict(x)

    # Calculate R-squared
    r_squared = r2_score(y, y_pred)
    lr_corrcoef = r_squared ** 0.5
    
    warnings.resetwarnings()
    return lr_corrcoef


print(f"Data samples: {len(prominence_df)}")
prominence_df = prominence_df.dropna(subset=[prominence_parameter])
print(f"Data samples after reduction: {len(prominence_df)}")

# make plots english
fig, axs = plt.subplots(2, 5, figsize=(14, 8))

emotions = ["neutral", "happy", "sad", "scared", "angry"]
surprisal_list = ['Surprisal GPT-2', 'Surprisal Yugo', 'Surprisal ngram-3']

surprisal_colour = {'Surprisal GPT-2': (0, 0 , 1, 1), 
                    'Surprisal Yugo':(1, 0 , 0, 1),
                    'Surprisal ngram-3':(1, 0, 1, 1)}

pp_title = {'time': 'Speech Time',
            'energy': 'RMS Energy',
            'f0': 'F0'}

fig.suptitle(f"Correlation Coefficient of Surprisal Values and {pp_title[prominence_parameter]}", fontsize=30)

for column_name in surprisal_list:
    for gender in ['f', 'm']:
        df_f = prominence_df[prominence_df['gender'] == gender]
        
        for idx, emotion in enumerate([0, 1, 2, 3, 4]):
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
            df_e = df_f[df_f['emotion'] == emotion]
            list_of_lr_values = []
            
            x = df_e[column_name].values
            y = df_e[prominence_parameter].values
            list_of_lr_values.append(calculate_corr_coef(x,y))
            
            list_of_columns = [column_name]
            for k in  range(1,11):
                list_of_columns.append(column_name + f" k={k}")
                df_e = df_e.dropna(subset=[column_name + f" k={k}"])
                
                x = df_e[list_of_columns]
                y = df_e[prominence_parameter].values
                list_of_lr_values.append(calculate_corr_coef(x,y))
            
            # Create x values from 0 to 10
            x_axis = list(range(11))
            if gender=='f' and emotion==0:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name], label=column_name[10:])
            else:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.plot(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)
    
# Add a common x-axis label
fig.text(0.5, 0.001, 'surprisal order', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, 'correlation coefficient', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()  
    

fig, axs = plt.subplots(2, 5, figsize=(14, 8))

surprisal_list = ['Surprisal BERT', 'Surprisal BERTic']

surprisal_colour = {'Surprisal BERT': (0, 0 , 1, 1), 
                    'Surprisal BERTic':(1, 0 , 0, 1)}

fig.suptitle(f"Correlation Coefficient of Surprisal Values and {pp_title[prominence_parameter]}", fontsize=30)

for column_name in surprisal_list:
    for gender in ['f', 'm']:
        df_f = prominence_df[prominence_df['gender'] == gender]
        
        for idx, emotion in enumerate([0, 1, 2, 3, 4]):
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
            df_e = df_f[df_f['emotion'] == emotion]
            list_of_lr_values = []
            
            x = df_e[column_name].values
            y = df_e[prominence_parameter].values
            list_of_lr_values.append(calculate_corr_coef(x,y))
            
            list_of_columns = [column_name]
            for k in  range(1,11):
                list_of_columns.append(column_name + f" k={k}")
                df_e = df_e.dropna(subset=[column_name + f" k={k}"])
                
                x = df_e[list_of_columns]
                y = df_e[prominence_parameter].values
                list_of_lr_values.append(calculate_corr_coef(x,y))
            
            # Create x values from 0 to 10
            x_axis = list(range(11))
            if gender=='f' and emotion==0:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name], label=column_name[10:])
            else:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.plot(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)
    
# Add a common x-axis label
fig.text(0.5, 0.001, 'surprisal order', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, 'correlation coefficient', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()  
    

# grafici na srpskom
fig, axs = plt.subplots(2, 5, figsize=(14, 8))

emotions = ["неутрално", "срећно", "тужно", "уплашено", "љуто"]
surprisal_list = ['Surprisal GPT-2', 'Surprisal Yugo', 'Surprisal ngram-3']

surprisal_colour = {'Surprisal GPT-2': (0, 0 , 1, 1), 
                    'Surprisal Yugo':(1, 0 , 0, 1),
                    'Surprisal ngram-3':(1, 0, 1, 1)}

pp_title = {'time': 'Времена Изговора',
            'energy': 'РМС Енергије',
            'f0': 'F0'}

fig.suptitle(f"Коефицијенти Корелације Вриједности Сурприсала и {pp_title[prominence_parameter]}", fontsize=30)

for column_name in surprisal_list:
    for gender in ['f', 'm']:
        df_f = prominence_df[prominence_df['gender'] == gender]
        
        for idx, emotion in enumerate([0, 1, 2, 3, 4]):
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
            df_e = df_f[df_f['emotion'] == emotion]
            list_of_lr_values = []
            
            x = df_e[column_name].values
            y = df_e[prominence_parameter].values
            list_of_lr_values.append(calculate_corr_coef(x,y))
            
            list_of_columns = [column_name]
            for k in  range(1,11):
                list_of_columns.append(column_name + f" k={k}")
                df_e = df_e.dropna(subset=[column_name + f" k={k}"])
                
                x = df_e[list_of_columns]
                y = df_e[prominence_parameter].values
                list_of_lr_values.append(calculate_corr_coef(x,y))
            
            # Create x values from 0 to 10
            x_axis = list(range(11))
            if gender=='f' and emotion==0:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name], label=column_name[10:])
            else:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.plot(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)
    
# Add a common x-axis label
fig.text(0.5, 0.001, 'ред сурприсала', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, 'коефицијент корелације', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()  
    

fig, axs = plt.subplots(2, 5, figsize=(14, 8))

surprisal_list = ['Surprisal BERT', 'Surprisal BERTic']

surprisal_colour = {'Surprisal BERT': (0, 0 , 1, 1), 
                    'Surprisal BERTic':(1, 0 , 0, 1)}

fig.suptitle(f"Коефицијенти Корелације Вриједности Сурприсала и {pp_title[prominence_parameter]}", fontsize=30)

for column_name in surprisal_list:
    for gender in ['f', 'm']:
        df_f = prominence_df[prominence_df['gender'] == gender]
        
        for idx, emotion in enumerate([0, 1, 2, 3, 4]):
            if gender == 'f':
                plt.subplot(2,5, emotion + 1)
            else:
                plt.subplot(2,5, emotion + 6)
            
            df_e = df_f[df_f['emotion'] == emotion]
            list_of_lr_values = []
            
            x = df_e[column_name].values
            y = df_e[prominence_parameter].values
            list_of_lr_values.append(calculate_corr_coef(x,y))
            
            list_of_columns = [column_name]
            for k in  range(1,11):
                list_of_columns.append(column_name + f" k={k}")
                df_e = df_e.dropna(subset=[column_name + f" k={k}"])
                
                x = df_e[list_of_columns]
                y = df_e[prominence_parameter].values
                list_of_lr_values.append(calculate_corr_coef(x,y))
            
            # Create x values from 0 to 10
            x_axis = list(range(11))
            if gender=='f' and emotion==0:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name], label=column_name[10:])
            else:
                plt.scatter(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.plot(x_axis, list_of_lr_values, color=surprisal_colour[column_name])
            
            plt.title(emotions[emotion], fontsize = 25)
            plt.tick_params(axis='both', which='major', labelsize=15)
    
# Add a common x-axis label
fig.text(0.5, 0.001, 'ред сурприсала', ha='center', va='center', fontsize=25)
# Add a common y-axis label
fig.text(0.0001, 0.5, 'коефицијент корелације', ha='center', va='center', rotation='vertical', fontsize=25)
fig.legend(fontsize=20, loc="center left", bbox_to_anchor=(1, 0.5))          
# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()  
    












