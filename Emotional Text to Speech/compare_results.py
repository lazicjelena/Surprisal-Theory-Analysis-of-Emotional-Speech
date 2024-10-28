# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:09:08 2024

@author: Jelena
"""

from sklearn.metrics import mean_squared_error
import pandas as pd 
import os

    
file_path = os.path.join('..','podaci','text-to-speech', 'liste', "Surprisal GPT-2_results.csv")
df = pd.read_csv(file_path)

model = 'Surprisal GPT-2 model'
#model = 'baseline model'

#print(f"Model: {model}")

for gender in ['f', 'm']:
    #gender_data = df[df['speaker gender']==gender]
    gender_data = df
    print(f"Gender: {gender}")
    
    for emotion in [0,1,2,3,4]:
        emotion_data = gender_data[gender_data['emotion']==emotion]
        
        emotion_data = emotion_data[['time', 'baseline model', model]].dropna()
        
        # Calculate MSE
        mse_model = mean_squared_error(emotion_data['time'].values, emotion_data[model])
        mse_baseline = mean_squared_error(emotion_data['time'].values, emotion_data['baseline model'])
        
        print(f"Emotion {emotion}:", mse_model - mse_baseline)
        
        
        