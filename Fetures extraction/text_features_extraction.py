# -*- coding: utf-8 -*-
"""text_features_extraction.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skripta prolazi kroz foldere i za svaki corrected transcript formira novi csv 
file na lokaciji data u kome cuva sva izdvojena obiljezja koja ce biit koristena 
u daljoj analizi i radu.

"""

import os
import pandas as pd
import warnings
from fuzzywuzzy import fuzz  

target_sentences_path = os.path.join('..','podaci', 'target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentences_path)

file_path = os.path.join('..','podaci', 'speaker_gender.txt')

try:
    # Read the text file into a DataFrame with custom column names
    speaker_gender_df = pd.read_csv(file_path, delimiter='\t', header=None, names=['combined'])

    # Extract speaker and gender using a regular expression
    speaker_gender_df[['speaker', 'gender']] = speaker_gender_df['combined'].str.extract(r'(\d+)\s*([mf])')

    # Drop the original combined column
    speaker_gender_df = speaker_gender_df.drop('combined', axis=1)

    print(speaker_gender_df)
except FileNotFoundError:
    print(f"The file at {file_path} could not be found.")
except pd.errors.EmptyDataError:
    print(f"The file at {file_path} is empty.")
except pd.errors.ParserError as pe:
    print(f"Error parsing the file at {file_path}: {pe}")
except Exception as e:
    print(f"An error occurred: {e}")

def find_target_sentence(sentence, df = target_sentences_df):
    max_similarity = 0
    target_index = -1  # Initialize with an invalid index

    for index, row in df.iterrows():
        current_similarity = fuzz.ratio(sentence, row['Text'])
        if current_similarity > max_similarity:
            max_similarity = current_similarity
            target_index = index

    return target_index

def get_word_position(word_count):
    # Determine the word position based on the word count
    if word_count == 1:
        return 'b'  # Beginning of the sentence
    elif word_count == 2:
        return 'm'  # Middle of the sentence
    else:
        return 'e'  # End of the sentence

def process_txt_file(file_path):
    # Add your code to process each TXT file here
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        index = find_target_sentence(lines[0][12:].strip())

        word_info_list = []
        total_words = 0
        for line in lines:
            if line.startswith('Word'):
                word_info = {}
                parts = line.split(', ')
                word_info['word'] = parts[0].split(': ')[1]
                word_info['start'] = float(parts[1].split(': ')[1])
                word_info['end'] = float(parts[2].split(': ')[1])

                # Determine word position in the sentence
                total_words += 1
                position = get_word_position(total_words)
                word_info['position'] = position
                word_info['target sentence'] = index

                word_info_list.append(word_info)

        return word_info_list

def calculate_word_length(word):

  words = word.split(' ')
  l = 0
  for i in words:
    l+= len(i)

  return l

def get_gender(speaker):
  return speaker_gender_df.loc[speaker_gender_df['speaker'] == speaker, 'gender'].values[0]

def process_directory(directory_path, output_path):

    # Create an empty DataFrame with desired column names
    columns = ['word', 'speaker', 'emotion', 'time', 'position', 'taget sentence']
    df = pd.DataFrame(columns=columns)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Extract information about the two last folders
        speaker = os.path.split(directory_path)[0].split(os.path.sep)[-2:][-1][:4]
        emotion = os.path.split(directory_path)[1]

        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            words = process_txt_file(file_path)

            # Add rows to the DataFrame for each word
            for word_info in words:
                word, start, end, position, target = word_info['word'], word_info['start'], word_info['end'], word_info['position'], word_info['target sentence']
                word_length = calculate_word_length(word)
                gender = get_gender(speaker)
                df = df.append({
                    'word': word,
                    'speaker': str(speaker),
                    'speaker gender': gender,
                    'emotion': emotion,
                    'length': word_length,
                    'time': float(end) - float(start),
                    'position': position,
                    'taget sentence': target
                }, ignore_index=True)

    # Create speaker and emotion folders if they don't exist
    speaker_folder = os.path.join(output_path, speaker)
    emotion_folder = os.path.join(speaker_folder, emotion)
    os.makedirs(emotion_folder, exist_ok=True)

    # Save the DataFrame as a CSV file in the emotion folder
    print('Output folder: ')
    print(emotion_folder)
    output_file_path = os.path.join(emotion_folder, speaker + '_' + emotion + '.csv')
    df.to_csv(output_file_path, index=False)

     # Reset warnings filter to default (optional)
    warnings.resetwarnings()

    return

directory_path = os.path.join('..','podaci', 'transcript_corrected')
output_directory = os.path.join('..','podaci', 'data')

# Loop through all folders in the main directory
for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
    print('Processing: ')
    print(folder_path)

    # Check if the current item is a directory
    if os.path.isdir(folder_path):
        # Loop through subdirectories within the current directory
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)

            # Check if the current item in the subdirectory is a directory
            if os.path.isdir(subfolder_path):
                process_directory(subfolder_path, output_directory)