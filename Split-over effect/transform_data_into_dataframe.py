# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

"""

import pandas as pd
import os

# Define the base directory
base_path = os.path.join('..','podaci','training data', 'general_data.csv') 
data = pd.read_csv(base_path)

# Add new columns for the length and log probability of the previous word
data['length -1'] = data['length'].shift(1)
data['log probability -1'] = data['log probability'].shift(1)
# Set values to NaN where target sentence and its shifted version are not the same
data.loc[data['target sentence'] != data['target sentence'].shift(1), ['length -1', 'log probability -1']] = pd.NA

for i in range(1,4):
    data[f"length -{i+1}"] = data[f"length -{i}"].shift(1)
    data[f"log probability -{i+1}"] = data[f"log probability -{i}"].shift(1)
    
    # Set to NaN where the previous columns were NaN
    data.loc[data[f"length -{i}"].isna(), f"length -{i+1}"] = pd.NA
    data.loc[data[f"log probability -{i}"].isna(), f"log probability -{i+1}"] = pd.NA

 
# Save the concatenated data to a CSV file
output_csv_path = os.path.join('..','podaci','split-over data', 'general_data.csv') 
data.to_csv(output_csv_path, index=False)
