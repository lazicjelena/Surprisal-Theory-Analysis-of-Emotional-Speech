# -*- coding: utf-8 -*-
"""transform_data_into_dataframe.py

Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Lag-feature builder for the surprisal-free baseline of the
split-over analysis. Reads
``../podaci/training data/general_data.csv`` (one row per spoken
word with at least ``length``, ``log probability``, and
``target sentence``) and adds five sentence-respecting lag
columns ``length -1 .. length -5`` and
``log probability -1 .. log probability -5`` by repeatedly
shifting; lag values that would cross a sentence boundary are
masked to NaN. The lagged table is saved to
``../podaci/split-over data/general_data.csv``, where
``Split-over effect/baseline_model.py`` consumes it to build the
per-lag ``baseline -k`` reference columns.
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
