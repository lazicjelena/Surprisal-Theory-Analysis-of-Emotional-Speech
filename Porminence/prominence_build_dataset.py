# -*- coding: utf-8 -*-
"""prominence_build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ova skripta sluzi da se napravi .csv file u kome su potpuno organizovani podaci
neophodni za racunanje korelacija i prikaz grafika koji se koriste u radu.
"""


import os
import pandas as pd

prosody_folder_path = os.path.join('..','podaci','prosody')

target_sentence_path = os.path.join('..','podaci', 'target_sentences.csv') 
target_sentence_df = pd.read_csv(target_sentence_path)

# Initialize an empty list to store DataFrames
speaker_list = []
emotion_list = []
word_list = []
prominence_list = []
boundary_list = []
target_sentence_list = []

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
                word = line_data[3]
                prominence = float(line_data[4])
                boundary = float(line_data[5])
                
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
    'target sentence': target_sentence_list
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
surprisal_bert_list = lookup_features(data, surprisal_bert, 'Surprisal BERT')
data['surprisal BERT'] = surprisal_bert_list
        
bertic_path = os.path.join('..','podaci', 'word_surprisals_bertic.csv') 
surprisal_bertic = pd.read_csv(bertic_path)
surprisal_bertic_list = lookup_features(data, surprisal_bertic, 'Surprisal BERTic')
data['surprisal BERTic'] = surprisal_bertic_list       
        
ngram3_path = os.path.join('..','podaci', 'word_surprisal_ngram3_alpha4.csv') 
surprisal_ngram3 = pd.read_csv(ngram3_path)
surprisal_ngram3_list = lookup_features(data, surprisal_ngram3, 'Surprisal ngram-3')
data['surprisal ngram3 alpha4'] = surprisal_ngram3_list

gpt_path = os.path.join('..','podaci', 'word_surprisals_gpt2.csv') 
surprisal_gpt = pd.read_csv(gpt_path)
surprisal_gpt_list = lookup_features(data, surprisal_gpt, 'Surprisal GPT-2')
data['surprisal GPT'] = surprisal_gpt_list

yugo_path = os.path.join('..','podaci', 'word_surprisals_yugo.csv') 
surprisal_yugo = pd.read_csv(yugo_path)
surprisal_yugo_list = lookup_features(data, surprisal_yugo, 'Surprisal Yugo')
data['surprisal yugo'] = surprisal_yugo_list

gender_info_path = os.path.join('..','podaci', 'gender_data.csv') 
gender_df = pd.read_csv(gender_info_path)
gender_df = gender_df.rename(columns={'Gender': 'gender', 'Speaker': 'speaker'})
data['speaker'] = data['speaker'].astype(int)
data = pd.merge(data, gender_df, on='speaker', how='left')

data = data[data['prominence']!=0]
output_csv_path = os.path.join('..','podaci', 'prominence_data.csv') 
data.to_csv(output_csv_path, index=False)





