# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:58:22 2024

@author: Jelena
"""

#pip install pydub

from pydub import AudioSegment
from pydub.effects import speedup
import pandas as pd
import numpy as np
import os

list_file_path = os.path.join('..','podaci','text-to-speech', 'liste', 'transformation_df.csv')
transformation_df = pd.read_csv(list_file_path)


def audio_duration_modification(file_path, target_duration):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)

    # Calculate original duration in seconds
    original_duration = len(audio) / 1000  # Convert milliseconds to seconds

    # Calculate the ratio of target to original duration
    speed_ratio = original_duration / target_duration

    if speed_ratio > 1:
        # Speed up the audio if the target duration is shorter
        modified_audio = audio.speedup(playback_speed=speed_ratio)
    else:
        # Slow down the audio if the target duration is longer
        modified_audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * speed_ratio)
        }).set_frame_rate(audio.frame_rate)

    return modified_audio


audio_folder_path = os.path.join('..','podaci','text-to-speech', 'Generated Words')
final_path = os.path.join('..','podaci','text-to-speech', 'GPT-2', 'Words')

for emotion in [0, 1, 2, 3, 4]:
  word_count = 0
  emotion_data = transformation_df[transformation_df['emotion']==emotion]

  # create emotion folder
  folder_path = final_path + str(emotion)
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")

  for _, row in emotion_data.iterrows():

    word = row['word']
    sentence = row['target_sentence']
    file_path = audio_folder_path + '/' + word + '.wav'
    target_duration = row['predicted time']
    
    if not np.isnan(target_duration):

        if target_duration > row['length']/20:
          modified_audio = audio_duration_modification(file_path, target_duration)
          modified_audio.export(final_path + str(emotion) + '/' + word + '_' +
                                str(sentence) + '.wav', format="wav")
          word_count +=1

  print(f'For emotion {emotion} number of modified words is {word_count}.')
  
''' Baseline model'''  
  
final_path = os.path.join('..','podaci','text-to-speech', 'Baseline model', 'Words')

for emotion in [0, 1, 2, 3, 4]:
  word_count = 0
  emotion_data = transformation_df[transformation_df['emotion']==emotion]

  # create emotion folder
  folder_path = final_path + str(emotion)
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")

  for _, row in emotion_data.iterrows():

    word = row['word']
    sentence = row['target_sentence']
    file_path = audio_folder_path + '/' + word + '.wav'
    target_duration = row['baseline model time']
    
    if not np.isnan(target_duration):

        if target_duration > row['length']/20:
          modified_audio = audio_duration_modification(file_path, target_duration)
          modified_audio.export(final_path + str(emotion) + '/' + word + '_' +
                                str(sentence) + '.wav', format="wav")
          word_count +=1

  print(f'For emotion {emotion} number of modified words is {word_count}.')