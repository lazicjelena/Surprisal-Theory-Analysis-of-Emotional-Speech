# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skripta prirpema skup podataka za analizu veze surprisala i mel koeficijenata.
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import pearsonr


def read_wav_files(directory):
    # List to store the paths of all WAV files
    wav_files = []

    # Iterate over all files in the directory
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        # Check if the file is a WAV file
        if file.endswith(".wav"):
            wav_files.append(file_path)

    return wav_files

def extract_transcript_info(transcript_path):
    try:
        # Read the transcript file
        with open(transcript_path, 'r', encoding='utf-8') as file:
            # Read only the first line of the file
            first_line = file.readline().strip()
        return first_line[12:]
    except FileNotFoundError:
        print(f"Transcript file not found at: {transcript_path}")
        return None

def read_transcript_files(audio_files):
    text_files = []
    for audio in audio_files:
      transcript_directory = audio.replace('data_mono', 'transcript_corrected')[:-4]
      transcript_path = os.path.join(transcript_directory + "_transcript.txt")
      try:
        # Extract transcript information
        transcript_info = extract_transcript_info(transcript_path)
      except FileNotFoundError:
        print(f"Transcript file not found for {audio}")
        transcript_info = ['']
      text_files.append(transcript_info)
    return text_files


def read_surprisal_values(text, word_surprisal_df, target_sentence_df):

  target_sentence_df['Text'] = target_sentence_df['Text'].str.lower()
  surprisal_values = []
  target_sentence_list = []
  for sentence in text:
    target_sentence = target_sentence_df[target_sentence_df['Text']==sentence.lower()]
    if not target_sentence.empty:
        target_sentence = target_sentence.index[0]
        surprisal_list = word_surprisal_df[word_surprisal_df['Sentence']==target_sentence]['Surprisal GPT-2'].tolist()
    else:
        target_sentence = None
        surprisal_list = [None]
    surprisal_values.append(surprisal_list)
    target_sentence_list.append(target_sentence)

  return surprisal_values, target_sentence_list


target_sentence_path = os.path.join('..','podaci', 'target_sentences.csv') 
target_sentence_df = pd.read_csv(target_sentence_path)

word_surprisal_path = os.path.join('..','podaci', 'word_surprisals_gpt2.csv') 
word_surprisal_df = pd.read_csv(word_surprisal_path )

# read audio files info
audio_path = os.path.join('..','podaci', 'data_mono') 
audio_files = []
speaker_list = []
emotion_list = []
text = []
# Traverse and process audio_path
for folder in os.listdir(audio_path):
    folder_path = os.path.join(audio_path, folder)
    if os.path.isdir(folder_path):
        # Loop through each inner folder (e.g., '0', '1', etc.)
        for inner_folder in os.listdir(folder_path):
            inner_folder_path = os.path.join(folder_path, inner_folder)  
            if os.path.isdir(inner_folder_path):
                single_speaker_list = read_wav_files(inner_folder_path)
                audio_files.append(single_speaker_list)
                text.append(read_transcript_files(single_speaker_list))
                speaker_list.append([folder for _ in range(0,len(single_speaker_list))])
                emotion_list.append([inner_folder for _ in range(0,len(single_speaker_list))])

# remove all files without corrected transcript
text = [item for sublist in text for item in sublist] 
                     
audio_files = [item for sublist in audio_files for item in sublist]  
audio_files = [audio_files[i] for i in range(len(text)) if text[i] is not None]

speaker_list = [item for sublist in speaker_list for item in sublist]    
speaker_list = [speaker_list[i] for i in range(len(text)) if text[i] is not None]

emotion_list = [item for sublist in emotion_list for item in sublist]  
emotion_list = [emotion_list[i] for i in range(len(text)) if text[i] is not None]

text = [text[i] for i in range(len(text)) if text[i] is not None]            
  
# read surprisal values                  
surprisal_values, target_sentence_list = read_surprisal_values(text, word_surprisal_df, target_sentence_df)

# remove all files with non target sentences
audio_files = [audio_files[i] for i in range(len(target_sentence_list)) if target_sentence_list[i] is not None]
speaker_list = [speaker_list[i] for i in range(len(target_sentence_list)) if target_sentence_list[i] is not None]
emotion_list = [emotion_list[i] for i in range(len(target_sentence_list)) if target_sentence_list[i] is not None]
text = [text[i] for i in range(len(target_sentence_list)) if target_sentence_list[i] is not None]    
surprisal_values = [surprisal_values[i] for i in range(len(target_sentence_list)) if target_sentence_list[i] is not None]  
target_sentence_list = [target_sentence_list[i] for i in range(len(target_sentence_list)) if target_sentence_list[i] is not None]  

# Keras tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# Define the file name
vocabulary_file_name = 'vocabulary_size.txt'
# Open the file in write mode, create it if it doesn't exist
with open(vocabulary_file_name, 'w') as file:
    # Write the vocabulary size to the file
    file.write(f"Vocabulary size keras tokenizer: {vocab_size}")

tokenized_texts = tokenizer.texts_to_sequences(text)
# Pad sequences
max_len = max(len(seq) for seq in tokenized_texts)
keras_padded_texts = pad_sequences(tokenized_texts, maxlen=max_len, padding='post')

# Random tokenization
tokens = set()
for sentence in text:
    tokens.update(sentence.split())

vocab_size = len(tokens) + 1
with open(vocabulary_file_name, 'w') as file:
    # Write the vocabulary size to the file
    file.write(f"Vocabulary size random tokenizer: {vocab_size}")

# Assign random indices to tokens
indices = np.random.permutation(len(tokens))
token_to_index = dict(zip(tokens, indices))

# Replace each token in the text with its random index
randomized_texts = []
for sentence in text:
    randomized_text = [token_to_index[token] for token in sentence.split()]
    randomized_texts.append(randomized_text)

tokenized_texts = randomized_texts

# Pad sequences
max_len = max(len(seq) for seq in tokenized_texts)
padded_texts = pad_sequences(tokenized_texts, maxlen=max_len, padding='post')

# Transpose surprisal values and pad
padded_surprisal_values = pad_sequences(surprisal_values, maxlen=max_len, padding='post', dtype='float32')


# Make folds for cross-validation
folds_path = os.path.join('..','podaci', 'folds.csv') 
folds_df = pd.read_csv(folds_path )
folds_list = [folds_df[folds_df['target sentence']==target].iloc[0]['fold'] for target in target_sentence_list]
 
   
# Convert to a DataFrame for easier handling
data = pd.DataFrame({
    'Audio Files': audio_files,
    'Text random': [seq.tolist() for seq in padded_texts],
    'Text keras':  [seq.tolist() for seq in keras_padded_texts],
    'Surprisal Values': [seq.tolist() for seq in padded_surprisal_values],
    'Speaker': speaker_list,
    'Emotion': emotion_list,
    'Target Sentence': target_sentence_list,
    'Fold': folds_list
})


# Save the DataFrame to a CSV file
output_csv_path = os.path.join('..','podaci', 'general_data.csv')
data.to_csv(output_csv_path, index=False)

# Check correlations between parameters
# Flatten the 2D arrays into 1D arrays
flattened_texts = np.concatenate([seq.flatten() for seq in padded_texts])
flattened_surprisal_values = np.concatenate([seq.flatten() for seq in padded_surprisal_values])
flattened_texts_keras = np.concatenate([seq.flatten() for seq in keras_padded_texts])

# Calculate the correlation
correlation, p_value = pearsonr(flattened_texts, flattened_surprisal_values)
print('Random tokenized text and surprisal')
print("Correlation:", correlation)
print("P-value:", p_value)

correlation, p_value = pearsonr(flattened_texts_keras, flattened_surprisal_values)
print('Keras tokenized text and surprisal')
print("Correlation:", correlation)
print("P-value:", p_value)
