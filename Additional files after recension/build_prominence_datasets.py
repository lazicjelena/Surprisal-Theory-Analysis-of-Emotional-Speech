# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:36:51 2025

@author: Jelena
"""

from my_functions import lookup_features, most_similar_sentence_index
import pandas as pd
import os


prosody_folder_path = os.path.join('..','podaci','prosody 1 0 0')
prosody = 'f0'

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
            index_value = most_similar_sentence_index(sentence, target_sentence_df)
                
            for i in range(0,len(lines)):
                target_sentence_list.append(index_value)

# Create DataFrame from the lists
data = pd.DataFrame({
    'speaker': speaker_list,
    'emotion': emotion_list,
    'word': word_list,
    f"{prosody}": prominence_list,
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


bert_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bert.csv') 
surprisal_bert = pd.read_csv(bert_path)
surprisal_bert_list = lookup_features(data, surprisal_bert,  'Surprisal BERT')
data['surprisal BERT'] = surprisal_bert_list
        
bertic_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_bertic.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal BERTic')
data['surprisal BERTic'] = surprisal_bertic_list       
        
ngram3_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisal_ngram3_alpha4.csv') 
surprisal_ngram3 = pd.read_csv(ngram3_path)
surprisal_ngram3_list = lookup_features(data, surprisal_ngram3, 'Surprisal ngram-3')
data['surprisal ngram3 alpha4'] = surprisal_ngram3_list

gpt_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_gpt2.csv') 
surprisal_gpt = pd.read_csv(gpt_path)
surprisal_gpt_list = lookup_features(data, surprisal_gpt, 'Surprisal GPT-2')
data['surprisal GPT'] = surprisal_gpt_list

yugo_path = os.path.join('..','podaci', 'surprisal values', 'word_surprisals_yugo.csv') 
surprisal_yugo = pd.read_csv(yugo_path)
surprisal_yugo_list = lookup_features(data, surprisal_yugo, 'Surprisal Yugo')
data['surprisal yugo'] = surprisal_yugo_list

gender_info_path = os.path.join('..','podaci', 'gender_data.csv') 
gender_df = pd.read_csv(gender_info_path)
gender_df = gender_df.rename(columns={'Gender': 'gender', 'Speaker': 'speaker'})
data['speaker'] = data['speaker'].astype(int)
data = pd.merge(data, gender_df, on='speaker', how='left')


fold_info_path = os.path.join('..','podaci', 'folds.csv') 
fold_df = pd.read_csv(fold_info_path)
data = pd.merge(data, fold_df, on='target sentence', how='left')


#data = data[data['energy']!=0]
output_csv_path = os.path.join('..','podaci', f"{prosody}_data.csv") 
data.to_csv(output_csv_path, index=False)