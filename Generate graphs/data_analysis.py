# -*- coding: utf-8 -*-
"""data_analysis.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skripta graficki predstavlja sve statisticke podatke prikazane u raud u poglavlju
analiza podataka.
"""

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import os


file_path =  os.path.join('..','podaci', 'concatenated_data.csv') 
df = pd.read_csv(file_path)

# Display basic information about the DataFrame
print(df.info())

# Display basic statistics about the numeric columns
print(df.describe())

# Count the frequency of values in the 'emotion' column
emotions = ['срећно', 'тужно', 'љуто', 'неутрално', 'уплашено']
emotion_counts_f = df[df['speaker gender'] == 'f']['emotion'].value_counts()
emotion_counts_m = df[df['speaker gender'] == 'm']['emotion'].value_counts()

# Set the width of the bars
bar_width = 0.3

# Define the x-axis locations for the groups
x_f = np.arange(len(emotions))
x_m = [x + bar_width for x in x_f]

# Plot the histograms
plt.bar(x_f, emotion_counts_f.values, width=bar_width, label='жене')
plt.bar(x_m, emotion_counts_m.values, width=bar_width, label='мушкарци')

# Set the x-axis labels
plt.xlabel('емоције', fontsize=15)
# Set the y-axis label
plt.ylabel('број ријечи', fontsize=15)
# Set the title
plt.title('Хистограм броја ријечи по емоционалним стањима', fontsize=15)
# Set the x-axis tick labels
plt.xticks([x + bar_width / 2 for x in range(len(emotions))], emotions)
# Add the legend
plt.legend()
# Show the plot
plt.show()

length_counts_f = df[df['speaker gender'] == 'f']['length'].value_counts()
length_counts_m = df[df['speaker gender'] == 'm']['length'].value_counts()

# Set the width of the bars
bar_width = 0.3

# Define the x-axis locations for the groups
x_f = np.arange(len(length_counts_f))
x_m = np.arange(len(length_counts_m)) + bar_width

# Plot the histograms
plt.bar(x_f, length_counts_f.values, width=bar_width, label='жене')
plt.bar(x_m, length_counts_m.values, width=bar_width, label='мушкарци')

plt.xlabel('дужина', fontsize=15)
plt.ylabel('дужина ријечи', fontsize=15)
plt.title('Хистограм ријечи по дужини', fontsize=15)


# Add the legend
plt.legend()
# Show the plot
plt.show()

# Count the frequency of values in the 'speaker gender' column
gender_counts = df['speaker gender'].value_counts()

# Plot the histogram
plt.bar(['жене', 'мушкарци'], gender_counts.values, color=['blue', 'orange'])
plt.xlabel('пол говорника')
plt.ylabel('број ријечи')
plt.title('Расподјела ријечи на основу пола говорника')
plt.show()

# Create a scatter plot of 'surprisal GPT' vs 'surprisal BERT'
plt.scatter(df['surprisal GPT'], df['surprisal BERT'])
plt.xlabel('Surprisal GPT')
plt.ylabel('Surprisal BERT')
plt.title('Correlation between Surprisal GPT and Surprisal BERT')
plt.show()

target_counts = df['target sentence'].value_counts()

# Plot the histogram
plt.bar(target_counts.index, target_counts.values)
plt.xlabel('Target Sentence')
plt.ylabel('Frequency')
plt.title('Frequency of Target Sentences')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

k = 10 # number of folds

# Assuming df is your DataFrame
unique_target_sentences = df['target sentence'].unique()

# Shuffle the list randomly
random.shuffle(unique_target_sentences)

numbers = list(range(0, len(unique_target_sentences)))
folds = [i%k for i in numbers]

folds_df = pd.DataFrame({
    'target sentence': unique_target_sentences,
    'fold': folds
})


# Merge the two DataFrames on 'target sentence'
merged_df = pd.merge(df, folds_df, on='target sentence', how='left')

# Count the frequency of values in the 'speaker gender' column
fold_counts = merged_df['fold'].value_counts()
# Save the concatenated data to a CSV file
output_folds_path = os.path.join('..','podaci', 'folds.csv') 
folds_df.to_csv(output_folds_path, index=False)

# Plot the histogram
plt.bar(fold_counts.index, fold_counts.values)
plt.xlabel('редни број фолда', fontsize=15)
plt.ylabel('број примјера', fontsize=15)
plt.title('Број примјера у сваком фолду', fontsize=15)
plt.show()


target_sentence_path =  os.path.join('..','podaci', 'target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentence_path)

for i in range(0,10):
 print(f"{target_sentences_df['Text'][i]}")