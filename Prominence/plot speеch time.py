# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 06:21:24 2024
@author: Jelena
lazic.jelenaa@gmail.com

Ova skripta prikazuje grafike promjene prozodijskih parametara za razlicita 
emocionalan stanja govornika.
"""

import pandas as pd
import os

data_path = os.path.join('..','podaci', 'prominence_data.csv') 
data = pd.read_csv(data_path)
columns_of_interest = ['speaker', 'emotion', 'word', 'target sentence', 'duration', 'gender', 'surprisal GPT']
data = data[columns_of_interest]

def extraxt_parameter_over_emotion(data, parameter):
    
    print(f"Parameter: {parameter}")
    neutral_data = data[data['emotion'] == 0]
    
    for emotion in [1,2,3,4]:
        
        print(f"Emotional state: {emotion}")
        duration_list = []
        none_values = 0
        ind = 0
        last_sentence = 932947234
        words = []
        
        for _,row in neutral_data.iterrows():
            
            speaker = row['speaker']
            sentence = row['target sentence']
            if sentence != last_sentence:
                last_sentence = sentence 
                words = []
            word = row['word']
            words.append(word)
            
            search = data[data['emotion']==emotion]
            search = search[search['target sentence']==sentence]
            search = search[search['speaker']==speaker]
            search = search[search['word']==word]
            
            if len(search) > 1:
                index = words.count(word)-1
                duration = search[parameter].values[index]
                ind += 1
            else:
                if len(search)==1:
                    duration = search[parameter].values[0]
                else:
                    duration = 487923472842101
                    none_values += 1
            duration_list.append(duration)
            
        neutral_data[emotion] = duration_list
        neutral_data = neutral_data[neutral_data[emotion] != 487923472842101]
        print(f"None values count: {none_values}")
        print(f"Double words count: {ind}")
        
    print(f"Final number of words: {len(neutral_data)}")
    
    return neutral_data

data_duration = extraxt_parameter_over_emotion(data, 'duration')

# Plot 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

emotions = ['neutral', 'happy', 'sad', 'scared', 'angry']
# Assuming data_duration is your DataFrame and emotion is a list of emotions [1,2,3,4]
fig, axs = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Spoken Word Duration Moulations over Different Emotional States', fontsize=30)

for idx, emotion in enumerate([1, 2, 3, 4]):
    # Plot for female speakers
    ax_f = axs[0, idx]
    df_f = data_duration[data_duration['gender'] == 'f']
    ax_f.scatter(df_f['duration'], df_f[emotion], color='r', label='Female')

    # Fit line and calculate MSE for female speakers
    coeffs_f = np.polyfit(df_f['duration'], df_f[emotion], 1)
    k_f, n_f = coeffs_f
    line_f = np.polyval(coeffs_f, df_f['duration'])
    mse_f = mean_squared_error(df_f[emotion], line_f)
    ax_f.plot(df_f['duration'], line_f, color='r', linestyle='--', 
              label=f'Fit: y={k_f:.2f}x+{n_f:.2f}\nMSE: {mse_f:.2f}')

    ax_f.set_title(f'{emotions[emotion]}', fontsize=25)
    ax_f.legend(fontsize=20)
    
    # Plot for male speakers
    ax_m = axs[1, idx]
    df_m = data_duration[data_duration['gender'] == 'm']
    ax_m.scatter(df_m['duration'], df_m[emotion], color='b', label='Male')

    # Fit line and calculate MSE for male speakers
    coeffs_m = np.polyfit(df_m['duration'], df_m[emotion], 1)
    k_m, n_m = coeffs_m
    line_m = np.polyval(coeffs_m, df_m['duration'])
    mse_m = mean_squared_error(df_m[emotion], line_m)
    ax_m.plot(df_m['duration'], line_m, color='b', linestyle='--', 
              label=f'Fit: y={k_m:.2f}x+{n_m:.2f}\nMSE: {mse_m:.2f}')

    ax_m.set_title(f'{emotions[emotion]}', fontsize=25)
    ax_m.legend(fontsize=20)

# Set x-axis label only in the middle of each row
for ax in axs[0, :]:
    ax.tick_params(axis='x', which='major', labelsize=20)

for ax in axs[1, :]:
    ax.tick_params(axis='x', which='major', labelsize=20)
    
# Increase the y-axis numbers' size for all subplots
for ax in axs.flat:
    ax.tick_params(axis='y', which='major', labelsize=20)  # Adjust labelsize as needed

# Add x-axis labels only in the middle
fig.text(0.5, 0.02, 'Neutral Speech Time [s]', ha='center', va='center', fontsize=25)
fig.text(0.002, 0.5, 'Emotional Speech time [s]', ha='center', va='center', rotation='vertical', fontsize=25)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Spoken Word Duration Modulations over Different Emotional States', fontsize=30)

for emotion in [1, 2, 3, 4]:
    # Female plots
    ax = fig.add_subplot(2, 4, emotion, projection='3d')
    df = data_duration[data_duration['gender'] == 'f']
    X = df[['duration', 'surprisal GPT']]
    y = df[emotion]
    
    # Fit the linear regression model (plane)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    ## Calculate MSE
    mse = mean_squared_error(y, y_pred)
    
    # Plot the points
    ax.scatter(df['duration'], df['surprisal GPT'], df[emotion], color='r', label=emotions[emotion])
    
    # Plot the plane
    xx, yy = np.meshgrid(np.linspace(X['duration'].min(), X['duration'].max(), 20),
                         np.linspace(X['surprisal GPT'].min(), X['surprisal GPT'].max(), 20))
    zz = model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy
    ax.plot_surface(xx, yy, zz, color='grey', alpha=0.5)
    
    ax.set_title(f'{emotions[emotion]}', fontsize=25)
    ax.set_xlabel('Time')
    ax.set_ylabel('Surprisal')
    ax.set_zlabel(f'Emotion Time')
    
    # Display plane parameters and MSE
    ax.text2D(0.05, 0.95, f'Plane: z={model.coef_[0]:.2f}x + {model.coef_[1]:.2f}y + {model.intercept_:.2f}\nMSE: {mse:.2f}',
              transform=ax.transAxes, fontsize=12, color='black')
    
    
    # Male plots
    ax = fig.add_subplot(2, 4, 4 + emotion, projection='3d')
    df = data_duration[data_duration['gender'] == 'm']
    X = df[['duration', 'surprisal GPT']]
    y = df[emotion]
    
    # Fit the linear regression model (plane)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate MSE
    mse = mean_squared_error(y, y_pred)
    
    # Plot the points
    ax.scatter(df['duration'], df['surprisal GPT'], df[emotion], color='b', label=emotions[emotion])
    
    # Plot the plane
    xx, yy = np.meshgrid(np.linspace(X['duration'].min(), X['duration'].max(), 20),
                         np.linspace(X['surprisal GPT'].min(), X['surprisal GPT'].max(), 20))
    zz = model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy
    ax.plot_surface(xx, yy, zz, color='grey', alpha=0.5)
    
    ax.set_title(f'{emotions[emotion]}', fontsize=25)
    ax.set_xlabel('Time')
    ax.set_ylabel('Surprisal')
    ax.set_zlabel(f'Emotion Time')
    
    # Display plane parameters and MSE
    ax.text2D(0.05, 0.95, f'Plane: z={model.coef_[0]:.2f}x + {model.coef_[1]:.2f}y + {model.intercept_:.2f}\nMSE: {mse:.2f}',
              transform=ax.transAxes, fontsize=12, color='black')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
