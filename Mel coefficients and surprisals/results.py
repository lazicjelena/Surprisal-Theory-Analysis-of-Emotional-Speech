# -*- coding: utf-8 -*-
"""results.py

@author: Jelena

Prikazivanje reuzultata veze surprisala i mel koeficijenata,
ovdje se radi i objedinjavanje rezultata u jedan skup podataka.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

gender = 'm'
emotion_name = ['happy', 'sad', 'angry', 'neutral', 'scarry']
directory_list = ['Results random', 'Results keras', 'Results surprisal']
emotions = [0, 1, 2, 3, 4]

# Initialize an empty list to store DataFrames
concatenated_data = []
loss_list = []

for directory in directory_list:
    directory_path = os.path.join('..','podaci','mel results', directory)
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Read each CSV file and append to the list of DataFrames
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(file_path)
        
        # Rename 'Test Loss' column
        loss_column_name = directory.split()[-1] + ' loss'
        df.rename(columns={'Test Loss': loss_column_name}, inplace=True)
        
        dataframes.append(df)
        
    # Concatenate all DataFrames into a single DataFrame
    concatenated_data.append(pd.concat(dataframes, ignore_index=True))
    loss_list.append(loss_column_name)
        

# Merge all DataFrames on 'Audio Files' column
data = concatenated_data[0]
for df in concatenated_data[1:]:
    data = pd.merge(data, df, on=['Audio Files', 'Emotion', 'Speaker'], how='outer')

# Merge the two DataFrames based on the Speaker column
gender_data_path = os.path.join('..','podaci','gender_data.csv')
gender_data = pd.read_csv(gender_data_path)
data = pd.merge(data, gender_data, on='Speaker', how='left') 
df = data[data['Gender']==gender]

print(f'Gender: {gender}')
for loss in loss_list:
    print(f'Parameter {loss}')
    for emotion in emotions:
        print(f'Emotion: {emotion}')
        emotion_df = df[df['Emotion']==emotion]
        mean_loss = emotion_df[loss].mean()
        std_loss = emotion_df[loss].std()
        print(f'Mean loss: {mean_loss}')
        print(f'Std: {std_loss}')


# Create a figure and axis object
fig, axes = plt.subplots(nrows=len(emotions), ncols=1, figsize=(10, 10), sharex=True)

# Iterate over each emotion
for idx, emotion in enumerate(emotions):
    # Filter the DataFrame for the current emotion
    df_emotion = df[df['Emotion'] == emotion]
    
    # Create a boxplot for each loss type
    sns.boxplot(data=df_emotion[loss_list], ax=axes[idx], width=0.5)
    
    # Add mean and standard deviation annotations
    for col in loss_list:
        mean = df_emotion[col].mean()
        std = df_emotion[col].std()
        axes[idx].annotate(f"Mean: {mean:.2f}\nStd: {std:.2f}", 
                           xy=(loss_list.index(col), mean), 
                           xytext=(10,7), 
                           textcoords='offset points',
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                     fc='yellow', 
                                     alpha=0.5))
    
    # Set title for each subplot
    axes[idx].set_title(f'Emotion {emotion_name[emotion]}')

# Add x-axis label
axes[-1].set_xlabel('Loss Type')

# Add y-axis label for all subplots
#fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
