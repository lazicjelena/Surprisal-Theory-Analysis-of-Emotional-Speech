# -*- coding: utf-8 -*-
"""word_frequency.py

Jelenina skripta
lazic.jelenaa@gmail.com

Prikaz grafika histograma rijeci, grafik je prikazan u radu.

"""

import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

train_data_path = os.path.join('..', '..','podaci','ngram_train_data.csv')
train = pd.read_csv(train_data_path)
# Convert all elements in the 'lemmas' column to strings
train['lemma'] = train['lemma'].astype(str)

# Create an empty Counter object to store word frequencies
word_freq = Counter()

# Iterate over each lemma in the 'lemma' column and update the word frequencies
for lemma in train['lemma']:
    # Assuming each lemma is a string containing space-separated words
    words = lemma.split()  # Split the lemma string into individual words
    word_freq.update(words)  # Update the word frequencies with the words from the lemma


# Filter word frequencies where frequency count is less than 100
filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq < 100}

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(filtered_word_freq.values(), bins=50, color='blue')
plt.title('Хистограм појављивања ријеч (учестаности < 100)', fontsize=16)
plt.xlabel('учестаност ријечи', fontsize=16)
plt.ylabel('број ријечи дате учестаности', fontsize=16)
plt.show()
