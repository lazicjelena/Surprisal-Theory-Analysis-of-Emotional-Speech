# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:05:43 2024

@author: Jelena
"""
import pandas as pd
import numpy as np
import os


ts_file_path = os.path.join('..','podaci', 'target_sentences.csv')
target_sentence_df = pd.read_csv(ts_file_path)

file_path = os.path.join('..','podaci','text-to-speech', 'liste', "Surprisal GPT-2_results.csv")
surprisal_df = pd.read_csv(file_path)

word_list = []
target_sentence_list = []
emotion_list = []
time_list = []
baseline_time_list = []
length_list = []

for index, row in target_sentence_df.iterrows():
    sentence = row['Text']
    
    for word in sentence.split():
        for emotion in [0, 1, 2, 3, 4]:
            
            emotion_list.append(emotion)
            target_sentence_list.append(index)
            word_list.append(word)
            length_list.append(len(word))
            
            df = surprisal_df[surprisal_df['word']==word]
            df = df[df['emotion']==emotion]
            df = df[df['target sentence']==index]
            
            if len(df)>0:
            
                time = df['Surprisal GPT-2 model'].values[0]
                time_list.append(time)
            
                baseline_time = df['baseline model'].values[0]
                baseline_time_list.append(baseline_time)
            
            else:
                time_list.append(np.nan)
                baseline_time_list.append(np.nan)

            
data = pd.DataFrame({
    'word': word_list,
    'length': length_list,
    'target_sentence': target_sentence_list,
    'emotion': emotion_list,
    'predicted time': time_list,
    'baseline model time': baseline_time_list
})



results_path = os.path.join('..','podaci','text-to-speech', 'liste', 'transformation_df.csv')
data.to_csv(results_path, index=False)





