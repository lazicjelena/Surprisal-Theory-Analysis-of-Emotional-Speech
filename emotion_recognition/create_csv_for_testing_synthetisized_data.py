# -*- coding: utf-8 -*-
"""create_csv_for_testing_synthetisized_data.py

Created on Wed Oct 23 16:09:41 2024

@author: Jelena

Pipeline role
-------------
Test-set CSV builder for the synthetic-speech evaluation of the
emotion-recognition model. Walks three sibling text-to-speech
output trees under ``../podaci/text-to-speech/`` (``GPT-2``,
``Sentences`` for Google TTS, and ``Baseline model``), and for
each WAV file records ``(file_path, label)`` where ``label`` is
recovered from the parent folder name (positions ``[10:]`` of the
top-level emotion-tagged folder; ``"0"`` for the unlabelled
Google TTS dump). The three resulting tables -
``data/surprisal_data.csv``, ``data/google_speech_data.csv`` and
``data/baseline_data.csv`` - are consumed by ``testing.py``.
"""

import pandas as pd
import os

# read training data
root_dir = os.path.join('..','podaci','text-to-speech', 'GPT-2', 'Sentences')

# Initialize lists to store file paths and their corresponding set type and last folder
train_files = []

# Walk through the directory
for main_folder in os.listdir(root_dir):
    main_folder_path = os.path.join(root_dir, main_folder)
    if os.path.isdir(main_folder_path):
        for file_name in os.listdir(main_folder_path):
            file_path = os.path.join(main_folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.wav'):
                # Store the file path along with the parent folder and last folder
                last_folder = main_folder[10:]
                file_info = (file_path, last_folder)
                train_files.append(file_info)
                        
# Convert the lists of tuples into DataFrames
data = pd.DataFrame(train_files, columns=['file_path', 'last_folder'])
data.to_csv(os.path.join('data','surprisal_data.csv'), index=False)

''' Google Speech Data '''

# read training data
root_dir = os.path.join('..','podaci','text-to-speech', 'Sentences')

# Initialize lists to store file paths and their corresponding set type and last folder
train_files = []

# Walk through the directory
for file_name in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.wav'):
        # Store the file path along with the parent folder and last folder
        last_folder = '0'
        file_info = (file_path, last_folder)
        train_files.append(file_info)
                        
# Convert the lists of tuples into DataFrames
data = pd.DataFrame(train_files, columns=['file_path', 'last_folder'])
data.to_csv(os.path.join('data','google_speech_data.csv'), index=False)

''' Baseline model time prediciton data - without surprisal'''

# read training data
root_dir = os.path.join('..','podaci','text-to-speech', 'Baseline model', 'Sentences')

# Initialize lists to store file paths and their corresponding set type and last folder
train_files = []

# Walk through the directory
for main_folder in os.listdir(root_dir):
    main_folder_path = os.path.join(root_dir, main_folder)
    if os.path.isdir(main_folder_path):
        for file_name in os.listdir(main_folder_path):
            file_path = os.path.join(main_folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.wav'):
                # Store the file path along with the parent folder and last folder
                last_folder = main_folder[10:]
                file_info = (file_path, last_folder)
                train_files.append(file_info)
                        
# Convert the lists of tuples into DataFrames
data = pd.DataFrame(train_files, columns=['file_path', 'last_folder'])
data.to_csv(os.path.join('data','baseline_data.csv'), index=False)