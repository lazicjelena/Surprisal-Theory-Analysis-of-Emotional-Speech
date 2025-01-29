# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 08:22:42 2024

@author: Jelena
"""

import numpy as np
import librosa
import pandas as pd
import os

word_list = []
user_list = []
gender_list = []
emotion_list = []
target_sentence_list = []
start_time_list = []
end_time_list = []
time_list = []
frequency_list = []
energy_list = []

# Define the directory path
folder_directory_path = os.path.join('..','podaci', 'data_mono')
transcript_directory_path = os.path.join('..','podaci', 'transcript_corrected')

gender_directory_path = os.path.join('..','podaci', 'gender_data.csv')
gender_df = pd.read_csv(gender_directory_path)
target_sentence_path = os.path.join('..','podaci', 'target_sentences.csv')
target_sentence_df = pd.read_csv(target_sentence_path)

fs = 44100

# Loop through the directory
for user in os.listdir(folder_directory_path):
    print(user)
    
    for emotion in [0,1,2,3,4]:
        directory_path = os.path.join(folder_directory_path, user, str(emotion))  
        
        if os.path.exists(directory_path):
            print(emotion)
            
            for filename in os.listdir(directory_path):
                if filename.endswith('.wav'):

                    file_path = os.path.join(directory_path, filename)
                    # Load the audio file
                    audio_file, sr = librosa.load(file_path, sr=fs)
                    transcript_path = os.path.join(transcript_directory_path, user, str(emotion), filename[:-4] + '_transcript.txt')
                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r', encoding='utf-8') as file:
                            first_line = file.readline().strip()
                            try:
                                index = target_sentence_df[target_sentence_df['Text'].str.lower() == first_line[12:]].index[0]
                            except:
                                index = 5347583875
                                
                            # Loop through the rest of the lines
                            for line in file:
                                line = line.strip()
                                
                                word = line.split(' ')[1][:-1]
                                ind = 2
                                while line.split(' ')[ind] != 'start:':
                                    word = ' '.join([word, line.split(' ')[ind][:-1]])
                                    ind += 1
                                word_list.append(word)
                                
                                user_list.append(user)
                                gender = gender_df[gender_df['Speaker'] == int(user)]['Gender'].values[0]
                                gender_list.append(gender)
                                emotion_list.append(emotion)
                                target_sentence_list.append(index)
                                
                                
                                start_time = float(line.split(' ')[line.split(' ').index('start:')+1][:-1])
                                start_time_list.append(start_time)
                                end_time = float(line.split(' ')[line.split(' ').index('end:')+1])
                                end_time_list.append(end_time)
                                time_list.append(end_time - start_time)
                                
                                y = audio_file[int(start_time*fs):int(end_time*fs)]
                                
                                if gender == 'm':
                                    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=90, fmax=155, sr=44100, frame_length=1024)
                                else:
                                    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=165, fmax=255, sr=44100, frame_length=1024)
                                frequency_list.append(np.nanmean(f0))
                                
                                energy = librosa.feature.rms(y=y, frame_length = 1024)
                                energy_list.append(np.nanmean(energy[0]))
                            
                      
  
# Create DataFrame
df = pd.DataFrame({'word': word_list,
                   'user': user_list,
                   'gender': gender_list,
                   'emotion': emotion_list,
                   'target sentence': target_sentence_list,
                   'start time': start_time_list,
                   'end time': end_time_list,
                   'time': time_list,
                   'frequency': frequency_list,
                   'energy': energy_list
                   })
# remove wrong transcriptions
df = df[df['target sentence']!=5347583875]

# Split conjoint words
def find_subword(word, unique_words):
    
    subword = ''
    for i in range(1,len(word)+1):
        if word[-i:] in unique_words:
            subword = word[-i:]
            
    return subword

unique_words = ' '.join(target_sentence_df['Text']).split()  
unique_words = set(word.lower() for word in unique_words)  

corrected_words = []

for word in df['word']:
    if word in unique_words:
        corrected_words.append(word)
    else:
        print(word)
        new_word = []
        while len(word)>0:
            subword = find_subword(word, unique_words)
            new_word.append(subword)
            word = word[:-len(subword)]
        new_word = ' '.join(new_word[::-1])
        print(new_word)
        corrected_words.append(new_word)
        
        
df['word'] = corrected_words

# add surprisal values
def lookup_features(data, freq_df, column_name):
    log_prob_list = []
    current_sentence = 1000
    list_of_words = []

    # Loop through rows of the DataFrame and print the 'word' column
    for index, row in data.iterrows():
        words = row['word'].split(' ')
        sentence = row['target sentence']
        if sentence != current_sentence:
          current_sentence = sentence
          list_of_words = []
        print(index)
        log_probability_value = 0
        for word in words:
            # Filter freq_df based on the 'Word' column
            freq_s = freq_df[freq_df['Sentence'] == sentence]
            freq = freq_s[freq_s['Word'] == word]

            # Extract the 'Log Probability' value for the filtered word
            #if not freq.empty:
            try:
                log_probability_value += freq[column_name].values[0 + list_of_words.count(word)]
            #else:
            except:
              log_probability_value += 0
              #print('error')
              #print(word)

            list_of_words.append(word)
            # avoid situation when two same sentences are one after another
            if len(list_of_words) == len(freq_s):
              list_of_words = []

        log_prob_list.append(log_probability_value)

    return log_prob_list


bert_path = os.path.join('..','podaci', 'word_surprisals_bert.csv') 
surprisal_bert = pd.read_csv(bert_path)
surprisal_bert_list = lookup_features(df, surprisal_bert, 'Surprisal BERT')
df['surprisal BERT'] = surprisal_bert_list
        
bertic_path = os.path.join('..','podaci', 'word_surprisals_bertic.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(df, surprisal_bertic, 'Surprisal BERTic')
df['surprisal BERTic'] = surprisal_bertic_list       
        
ngram3_path = os.path.join('..','podaci', 'word_surprisal_ngram3_alpha4.csv') 
surprisal_ngram3 = pd.read_csv(ngram3_path)
surprisal_ngram3_list = lookup_features(df, surprisal_ngram3, 'Surprisal ngram-3')
df['surprisal ngram3 alpha4'] = surprisal_ngram3_list

gpt_path = os.path.join('..','podaci', 'word_surprisals_gpt2.csv') 
surprisal_gpt = pd.read_csv(gpt_path)
surprisal_gpt_list = lookup_features(df, surprisal_gpt, 'Surprisal GPT-2')
df['surprisal GPT'] = surprisal_gpt_list

yugo_path = os.path.join('..','podaci', 'word_surprisals_yugo.csv') 
surprisal_yugo = pd.read_csv(yugo_path)
surprisal_yugo_list = lookup_features(df, surprisal_yugo, 'Surprisal Yugo')
df['surprisal yugo'] = surprisal_yugo_list

# save the data
final_path = os.path.join('..','podaci', 'prosody_surprisal.csv')
df.to_csv(final_path, index=False)  # Saves the DataFrame as a CSV file