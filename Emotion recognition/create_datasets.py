# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:20:51 2024

@author: Jelena
"""

from audiodataset import create_dataloader
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# read training data
root_dir = os.path.join('..','podaci','data_mono')


# Initialize lists to store file paths and their corresponding set type and last folder
train_files = []

# Walk through the directory
for main_folder in os.listdir(root_dir):
    main_folder_path = os.path.join(root_dir, main_folder)
    if os.path.isdir(main_folder_path):
        for sub_folder in os.listdir(main_folder_path):
            sub_folder_path = os.path.join(main_folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):
                for file_name in os.listdir(sub_folder_path):
                    file_path = os.path.join(sub_folder_path, file_name)
                    if os.path.isfile(file_path) and file_name.endswith('.wav'):
                        # Store the file path along with the parent folder and last folder
                        parent_folder = main_folder
                        last_folder = sub_folder
                        file_info = (file_path, last_folder)
                        train_files.append(file_info)
                        
# Convert the lists of tuples into DataFrames
data = pd.DataFrame(train_files, columns=['file_path', 'last_folder'])

# Define the split ratios
train_ratio = 0.65
val_ratio = 0.20
test_ratio = 0.15

# Split train_df into train and temp (validation + test) with stratification by 'last_folder'
train_df, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42, stratify=data['last_folder'])

# Further split temp into validation and test sets with stratification by 'last_folder'
val_df, test_df = train_test_split(temp_data, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42, stratify=temp_data['last_folder'])

train_df.to_csv(os.path.join('data','train_dataset.csv'), index=False)
print(f'Training data: {len(train_df)}')
val_df.to_csv(os.path.join('data','val_dataset.csv'), index=False)
print(f'Validation data: {len(val_df)}')
test_df.to_csv(os.path.join('data','test_dataset.csv'), index=False)
print(f'Test data: {len(test_df)}')
