# -*- coding: utf-8 -*-
"""plot frequency.py

Created on Mon Aug 19 10:57:31 2024

@author: Jelena

Pipeline role
-------------
Frequency-modulation plotting script for the ``Prominence/``
analysis. Identical structure to ``plot energy.py``: reads
``../podaci/prominence_data.csv`` (built by
``prominence_build_dataset.py``), aligns each emotional row
with its neutral counterpart via
:func:`analysis_utils.extraxt_parameter_over_emotion` on
``"prominence"`` and draws the same 2x4 per-emotion / per-gender
scatter + linear fit grid. Output is purely visual; no CSVs are
written. (No 3-D regression-plane figure is rendered in this
file - that part of the comparison sits in ``plot energy.py``
and ``plot speech time.py``.)
"""

import pandas as pd
import os

from utils.analysis_utils import extraxt_parameter_over_emotion

data_path = os.path.join('..','podaci', 'prominence_data.csv')
data = pd.read_csv(data_path)
columns_of_interest = ['speaker', 'emotion', 'word', 'target sentence', 'duration', 'prominence', 'gender', 'surprisal GPT']
data = data[columns_of_interest]

data_duration = extraxt_parameter_over_emotion(data, 'prominence')

# Plot 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

emotions = ['neutral', 'happy', 'sad', 'scared', 'angry']
# Assuming data_duration is your DataFrame and emotion is a list of emotions [1,2,3,4]
fig, axs = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Frequency Modulations over Different Emotional States', fontsize=30)

for idx, emotion in enumerate([1, 2, 3, 4]):
    # Plot for female speakers
    ax_f = axs[0, idx]
    df_f = data_duration[data_duration['gender'] == 'f']
    ax_f.scatter(df_f['prominence'], df_f[emotion], color='r', label='Female')

    # Fit line and calculate MSE for female speakers
    coeffs_f = np.polyfit(df_f['prominence'], df_f[emotion], 1)
    k_f, n_f = coeffs_f
    line_f = np.polyval(coeffs_f, df_f['prominence'])
    mse_f = mean_squared_error(df_f[emotion], line_f)
    ax_f.plot(df_f['prominence'], line_f, color='r', linestyle='--', 
              label=f'Fit: y={k_f:.2f}x+{n_f:.2f}\nMSE: {mse_f:.2f}')

    ax_f.set_title(f'{emotions[emotion]}', fontsize=25)
    ax_f.legend(fontsize=20)
    
    # Plot for male speakers
    ax_m = axs[1, idx]
    df_m = data_duration[data_duration['gender'] == 'm']
    ax_m.scatter(df_m['prominence'], df_m[emotion], color='b', label='Male')

    # Fit line and calculate MSE for male speakers
    coeffs_m = np.polyfit(df_m['prominence'], df_m[emotion], 1)
    k_m, n_m = coeffs_m
    line_m = np.polyval(coeffs_m, df_m['prominence'])
    mse_m = mean_squared_error(df_m[emotion], line_m)
    ax_m.plot(df_m['prominence'], line_m, color='b', linestyle='--', 
              label=f'Fit: y={k_m:.2f}x+{n_m:.2f}\nMSE: {mse_m:.2f}')

    ax_m.set_title(f'{emotions[emotion]}', fontsize=25)
    ax_m.legend(fontsize=20)

# Set x-axis label only in the middle of each row
for ax in axs[0, :]:
    ax.tick_params(axis='x', which='major', labelsize=20)

for ax in axs[1, :]:
    ax.tick_params(axis='x', which='major', labelsize=20)
    
# Increase the y-axis numbers' size for all subplots
for ax in axs.flat:
    ax.tick_params(axis='y', which='major', labelsize=20)  # Adjust labelsize as needed

# Add x-axis labels only in the middle
fig.text(0.5, 0.02, 'Neutral Speech', ha='center', va='center', fontsize=25)
fig.text(0.02, 0.5, 'Emotional Speech', ha='center', va='center', rotation='vertical', fontsize=25)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
