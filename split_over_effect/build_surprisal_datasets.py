# -*- coding: utf-8 -*-
"""build_surprisal_datasets.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Per-surprisal lag-feature builder for the split-over analysis.
For each surprisal channel in
``['Surprisal GPT-2', 'Surprisal Yugo', 'Surprisal BERT',
'Surprisal BERTic', 'Surprisal ngram-3']``, reads
``../podaci/training data/<surprisal>.csv`` (one row per spoken
word with the surprisal value already attached), then adds five
lag columns ``-1 .. -5`` by repeatedly shifting the surprisal
column. The lagged table is saved alongside as
``../podaci/split-over data/<surprisal>.csv``, where it is later
joined with the lag-aware baseline by ``surprisal_results.py``.

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




    
    