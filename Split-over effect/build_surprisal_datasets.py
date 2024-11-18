# -*- coding: utf-8 -*-
"""build_dataset.py

Jelenina skripta
lazic.jelenaa@gmail.com

"""

import pandas as pd
import os

surprisal_name = ['Surprisal GPT-2',
                  'Surprisal Yugo', 
                  'Surprisal BERT',
                  'Surprisal BERTic',
                  'Surprisal ngram-3'
                  ]

for surprisal in surprisal_name:
    
    df_path = os.path.join('..','podaci', 'training data', f"{surprisal}.csv") 
    data = pd.read_csv(df_path)
    
    # Add new columns for the length and log probability of the previous word
    data[f"{surprisal} -1"] = data[f"{surprisal}"].shift(1)
    
    for i in range(1,5):
        data[f"{surprisal} -{i+1}"] = data[f"{surprisal} -{i}"].shift(1)
    
    # Save the concatenated data to a CSV file
    output_csv_path = os.path.join('..','podaci','split-over data', f"{surprisal}.csv") 
    data.to_csv(output_csv_path, index=False)




    
    