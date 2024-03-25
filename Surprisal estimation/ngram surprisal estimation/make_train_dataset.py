# -*- coding: utf-8 -*-
"""make_train_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skripta objedinjuje sve skupove podataka na kojima se odredjuje ngram model u 
jedan .csv fajl. Prije toga podaci su izdvojeni, formirane su recenice i izvrsena
je lematizacija.

"""

import pandas as pd
import os

# Directory containing CSV files
directory = os.path.join('..', '..', 'podaci','ngram datasets')

# List to hold all dataframes
dfs = []

# Read each CSV file in the directory
for file_path in os.listdir(directory):
    if file_path.endswith(".csv"):
        filename = os.path.join(directory, file_path)
        df = pd.read_csv(filename)
        dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Remove duplicates
unique_df = combined_df.drop_duplicates()

# Save unique data to a new CSV file
output_file = os.path.join('..', '..', 'podaci','ngram_train_data.csv')
unique_df.to_csv(output_file, index=False)

print("Unique data saved to", output_file)
