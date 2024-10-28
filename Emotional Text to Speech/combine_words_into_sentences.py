# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:45:46 2024

@author: Jelena
"""

from pydub import AudioSegment
import pandas as pd
import os


ts_file_path = os.path.join('..','podaci', 'target_sentences.csv')
target_sentences = pd.read_csv(ts_file_path)

audio_path = os.path.join('..','podaci','text-to-speech', 'Generated Words')
final_path = os.path.join('..','podaci','text-to-speech', 'Sentences')

for index, row in target_sentences.iterrows():

  sentence = row['Text'].lower()
  words = sentence.split(' ')

  # Initialize an empty AudioSegment object
  combined_audio = AudioSegment.empty()

  for word in words:
    word_path = audio_path +  "/" + word + '.wav'
    if os.path.exists(word_path):
        word_audio = AudioSegment.from_wav(word_path)
        combined_audio += word_audio
    else:
        print(f"Audio file for word '{word}' not found.")

  output_path = os.path.join(final_path, sentence + '.wav')
  combined_audio.export(output_path, format='wav')
 
''' Generate emotional sentences based on surprisal model'''

audio_path = os.path.join('..','podaci','text-to-speech', 'GPT-2')
final_path = os.path.join('..','podaci','text-to-speech', 'GPT-2', 'Sentences')  

for emotion in [0, 1, 2, 3, 4]:
  for index, row in target_sentences.iterrows():
    sentence = row['Text'].lower()
    words = sentence.split(' ')

    # Initialize an empty AudioSegment object
    combined_audio = AudioSegment.empty()

    for word in words:
      word_path = audio_path + f'/Words{emotion}/{word}_{index}.wav'
      if os.path.exists(word_path):
        word_audio = AudioSegment.from_wav(word_path)
        combined_audio += word_audio
      else:
        word_path = f'..\\podaci\\text-to-speech\\Generated Words\\{word}.wav'
        if os.path.exists(word_path):
          word_audio = AudioSegment.from_wav(word_path)
          combined_audio += word_audio
        else:
          print(f"Audio file for word '{word}' not found.")

    # Export the combined audio
    # create emotion folder
    final_folder_path = final_path + f'/Sentences {emotion}/'
    if not os.path.exists(final_folder_path):
      os.makedirs(final_folder_path)
      print(f"Folder '{final_folder_path}' created.")

    output_path = os.path.join(final_folder_path, sentence + '.wav')
    combined_audio.export(output_path, format='wav')
    
''' Generate emotional sentences based on baseline model'''

audio_path = os.path.join('..','podaci','text-to-speech', 'Baseline model')
final_path = os.path.join('..','podaci','text-to-speech', 'Baseline model', 'Sentences')  

for emotion in [0, 1, 2, 3, 4]:
  for index, row in target_sentences.iterrows():
    sentence = row['Text'].lower()
    words = sentence.split(' ')

    # Initialize an empty AudioSegment object
    combined_audio = AudioSegment.empty()

    for word in words:
      word_path = audio_path + f'/Words{emotion}/{word}_{index}.wav'
      if os.path.exists(word_path):
        word_audio = AudioSegment.from_wav(word_path)
        combined_audio += word_audio
      else:
        word_path = f'..\\podaci\\text-to-speech\\Generated Words\\{word}.wav'
        if os.path.exists(word_path):
          word_audio = AudioSegment.from_wav(word_path)
          combined_audio += word_audio
        else:
          print(f"Audio file for word '{word}' not found.")

    # Export the combined audio
    # create emotion folder
    final_folder_path = final_path + f'/Sentences {emotion}/'
    if not os.path.exists(final_folder_path):
      os.makedirs(final_folder_path)
      print(f"Folder '{final_folder_path}' created.")

    output_path = os.path.join(final_folder_path, sentence + '.wav')
    combined_audio.export(output_path, format='wav')