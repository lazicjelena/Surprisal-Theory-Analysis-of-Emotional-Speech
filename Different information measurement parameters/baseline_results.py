# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:02:57 2024

@author: Jelena
"""

from my_functions import add_column
import pandas as pd
import numpy as np
import os
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# suprrisal, entropy, information values, adjusted surrpisal
#file_path = os.path.join('..','podaci','information measurements parameters', "data.csv") 

# non-context embedding
#file_path = os.path.join('..','podaci','information measurements parameters', "non_context_embedding_data.csv")
#file_path = os.path.join('..','podaci','information measurements parameters', "non_context_embedding_data_surprisal.csv")

# context embedding
#file_path = os.path.join('..','podaci','information measurements parameters', "context_embedding_data.csv")
file_path = os.path.join('..','podaci','information measurements parameters', "context_embedding_data_surprisal.csv")

df = pd.read_csv(file_path)
df = df.replace("nan", np.nan)
df = df[df['time']!=0]

results_df = add_column(df, 3)    
df = pd.merge(df, results_df, how='left')

df.to_csv(file_path, index=False)

