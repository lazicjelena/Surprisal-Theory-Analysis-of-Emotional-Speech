# -*- coding: utf-8 -*-
"""surprisal_estimation_ngram_model.py

Jelenina skripta
lazic.jelenaa@gmail.com

Racunanje surprisal rijeci na nivou ngrama, za racunanje n= 2,3,4,5 potrebno
je promijeniti vrijednost promjenjive u kodu. Ovdje se moze mijenjati i parametar alpha.


"""

import pandas as pd
import nltk
from nltk import ngrams
from collections import Counter
from nltk.util import pad_sequence
import os
import math

num_grams = 2
alpha = 4 # regularization parameter

def calculate_vocabulary_size(data):
    # Create a set to store unique words
    unique_words = set()

    # Iterate over each sentence in the data
    for sentence in data:
        # Add each word to the set of unique words
        unique_words.update(sentence)

    # Return the size of the set, which is the vocabulary size
    return len(unique_words)

def calculate_word_probabilities(sentence, n_gram_counts, vocabulary_size, n=3):
    
    
    # Generate n-grams from the sentence
    n_grams = list(ngrams(sentence, n))

    # Calculate the probability for each word
    probabilities = []
    for n_gram in n_grams:

        # Calculate the count of the n-gram and the count of the (n-1)-gram
        n_gram_count = n_gram_counts.get(n_gram, 0)
        n_gram_count_1 = n_gram_counts_1.get(n_gram[:-1], 0)

        # Calculate the Laplace smoothed probability
        probability = (n_gram_count + alpha) / (n_gram_count_1 + alpha * vocabulary_size)
        probabilities.append((n_gram[-1], probability))
        
    return probabilities

train_data_path = os.path.join('..', '..', 'podaci','ngram_train_data.csv')
train = pd.read_csv(train_data_path)
# Convert all elements in the 'lemmas' column to strings
train['lemmas'] = train['lemma'].astype(str)

test_data_path = os.path.join('..', '..', 'podaci','target_sentences_lemmas.csv')
test = pd.read_csv(test_data_path)
test['lemmas'] = test['lemma'].astype(str)


# Tokenize the sentences into words
tokenized_sentences = [list(pad_sequence(nltk.word_tokenize(sentence),pad_left=True, left_pad_symbol="<s>",
                  pad_right=True, right_pad_symbol="</s>", n=num_grams)) for sentence in train['lemmas']]

# Flatten the list of tokenized sentences
flat_words = []
for sentence in tokenized_sentences:
    for word in sentence:
        if word not in [',', '.', '!', '?']:
            flat_words.append(word)

# Generate n-grams
n_grams = list(ngrams(flat_words, num_grams))
n_grams_1 = list(ngrams(flat_words, num_grams-1))

# Count the occurrences of each n-gram
n_gram_counts = Counter(n_grams)
n_gram_counts_1 = Counter(n_grams_1)

# Print the most common n-grams
print("Most common n-grams:")
for n_gram, count in n_gram_counts.most_common(10):
    print(f"{n_gram}: {count}")

# Print the most common n-grams
print("Most common n-1-grams:")
for n_gram_1, count in n_gram_counts_1.most_common(10):
    print(f"{n_gram_1}: {count}")

# Assuming you have a list of tokenized sentences named 'tokenized_sentences'
vocabulary_size = calculate_vocabulary_size(tokenized_sentences)
print("Vocabulary size:", vocabulary_size)

def flatten_sentence_processing(sentence):
    
    # Pad the sentence
    padded_sentence = list(pad_sequence(nltk.word_tokenize(sentence), pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=num_grams))
    flatten_sentence = []
    for word in padded_sentence:
        if word not in [',', '.', '!', '?']:
            flatten_sentence.append(word)
    return flatten_sentence

# test model
sentence_list = []
word_list = []
surprisal_list = []
lema_list = []

for i in range(0,len(test)):
    sentence = test['lemmas'][i]
    flatten_sentence = flatten_sentence_processing(sentence)
    word_sentence = flatten_sentence_processing(test['Text'][i])
    # Calculate word probabilities for the padded sentence
    result = calculate_word_probabilities(flatten_sentence, n_gram_counts,vocabulary_size, n=num_grams)
    
    j = num_grams-1
    for lema,probability in result:
        if lema not in ["</s>","<s>"]:
            sentence_list.append(i)
            lema_list.append(lema)
            word_list.append(word_sentence[j].lower())
            surprisal_list.append(-math.log2(probability))
            j+=1

df = pd.DataFrame({'Sentence': sentence_list, 'Word': word_list, 'Surprisal ngram-3': surprisal_list})
# Save the DataFrame to a CSV file
define_name = f'word_surprisal_ngram{num_grams}_alpha{alpha}.csv'
csv_file_path = os.path.join('..', '..', 'podaci', define_name)
df.to_csv(csv_file_path, index=False)
