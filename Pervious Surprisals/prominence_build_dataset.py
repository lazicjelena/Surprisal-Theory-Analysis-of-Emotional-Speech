# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 05:59:56 2024

@author: Jelena
"""


import os
import pandas as pd

prosody_folder_path = os.path.join('..','podaci','prosody 1 0 0') # frequency
prosody_folder_path = os.path.join('..','podaci','prosody') # energy

target_sentence_path = os.path.join('..','podaci', 'target_sentences.csv') 
target_sentence_df = pd.read_csv(target_sentence_path)

# Initialize an empty list to store DataFrames
speaker_list = []
emotion_list = []
word_list = []
prominence_list = []
boundary_list = []
target_sentence_list = []
duration_list = []

# Iterate through each file in the folder
for file_name in os.listdir(prosody_folder_path):
    # Check if the file is a .prom file
    if file_name.endswith('.prom'):
        # Extract speaker, emotion from file name
        speaker = file_name[:4]
        emotion = file_name[5] 
        # Read the content of the .prom file
        with open(os.path.join(prosody_folder_path, file_name), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            sentence = []
            # Extract data from each line and append to DataFrame
            for line in lines:
                line_data = line.strip().split('\t')
                duration = float(line_data[2]) - float(line_data[1])
                word = line_data[3]
                prominence = float(line_data[4])
                boundary = float(line_data[5])
                
                duration_list.append(duration)
                sentence.append(word)
                speaker_list.append(speaker)
                emotion_list.append(emotion)
                word_list.append(word)
                prominence_list.append(prominence)
                boundary_list.append(boundary)
                
                
            # Create a sentence by joining the words with spaces
            sentence = ''.join(sentence)    
            # Initialize index variable
            index = None

            # Iterate through the DataFrame
            for idx, row in target_sentence_df.iterrows():
                # Check if the text in the row matches the target sentence
                if row['Text'].replace(' ', '') == sentence:
                    # If matched, assign the index and break the loop
                    index = idx
                    break
                
            for i in range(0,len(lines)):
                target_sentence_list.append(index)

# Create DataFrame from the lists
data = pd.DataFrame({
    'speaker': speaker_list,
    'emotion': emotion_list,
    'word': word_list,
    'prominence': prominence_list,
    'boundary': boundary_list,
    'target sentence': target_sentence_list,
    'duration': duration_list
})


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

for word in data['word']:
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
          
data['word'] = corrected_words

# save data
data = data.rename(columns={"prominence": "energy"})
output_csv_path = os.path.join('..','podaci', 'correlation data','energy_prominence_data.csv') 
data.to_csv(output_csv_path, index=False)


# merge prominence parameters
f0_csv_path = os.path.join('..','podaci', 'correlation data','f0_prominence_data.csv') 
energy_csv_path = os.path.join('..','podaci', 'correlation data','energy_prominence_data.csv') 

f0_data = pd.read_csv(f0_csv_path)
energy_data = pd.read_csv(energy_csv_path)

data = pd.merge(f0_data, energy_data, how='left')
data = data.rename(columns={"duration": "time"})

gender_path =   os.path.join('..','podaci', 'gender_data.csv') 
gender_df = pd.read_csv(gender_path)
gender_df = gender_df.rename(columns={"Speaker":"speaker", "Gender":"gender"})
data = pd.merge(data, gender_df, on='speaker', how='left')

output_csv_path = os.path.join('..','podaci', 'correlation data','prominence_data.csv') 
data.to_csv(output_csv_path, index=False)

