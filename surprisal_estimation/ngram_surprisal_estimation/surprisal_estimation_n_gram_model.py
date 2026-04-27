# -*- coding: utf-8 -*-
"""surprisal_estimation_n_gram_model.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Trains an ``n``-gram language model with Laplace (additive)
smoothing on the lemmatised training corpus
``../../podaci/ngram_train_data.csv`` (produced by
``make_train_dataset.py``) and uses it to estimate per-word
surprisal for the canonical target sentences
``../../podaci/target_sentences_lemmas.csv``. Tunable knobs are
the module-level ``num_grams`` (n) and ``alpha``
(smoothing parameter). The five most-common n-grams and
(n-1)-grams are dropped from the count tables (a heuristic to
deweight punctuation-only or boundary-only n-grams). For each
target sentence the script tokenises and pads with ``<s>`` /
``</s>``, computes the per-word smoothed probability, and emits
``-log2(p)`` as the surprisal. Result CSV is written as
``../../podaci/word_surprisal_ngram<n>_alpha<alpha>.csv`` (e.g.
``word_surprisal_ngram3_alpha4.csv`` with the defaults), which is
the input to ``Pervious Surprisals/build_dataset.py`` (column
``Surprisal ngram-3``).

"""

import pandas as pd
import nltk
from nltk import ngrams
from collections import Counter
from nltk.util import pad_sequence
import os
import math

num_grams = 3
alpha = 4 # regularization parameter

def calculate_vocabulary_size(data):
    """Return the size of the unique-token vocabulary across ``data``.

    Iterates over each tokenised sentence in ``data`` and updates a
    running ``set`` with the tokens, then returns its size. The
    returned vocabulary size ``V`` is used as the denominator
    coefficient ``alpha * V`` in the Laplace-smoothed n-gram
    probability used in
    :func:`calculate_word_probabilities`.

    Parameters
    ----------
    data : iterable of iterable of str
        One tokenised sentence per element. Tokens are added to
        the vocabulary as-is (no lower-casing, no stripping).

    Returns
    -------
    int
        Number of unique tokens across all sentences.
    """
    # Create a set to store unique words
    unique_words = set()

    # Iterate over each sentence in the data
    for sentence in data:
        # Add each word to the set of unique words
        unique_words.update(sentence)

    # Return the size of the set, which is the vocabulary size
    return len(unique_words)

def calculate_word_probabilities(sentence, n_gram_counts, vocabulary_size, n=3):
    """Score every ``n``-gram in ``sentence`` with Laplace smoothing.

    Generates the list of overlapping ``n``-grams of ``sentence``
    and, for each one, computes the additive-smoothed conditional
    probability ``(c(n_gram) + alpha) / (c(prefix) + alpha * V)``
    where ``c(...)`` are the counts in ``n_gram_counts`` and
    ``n_gram_counts_1`` (module-level), ``alpha`` is the module-level
    smoothing constant and ``V`` is ``vocabulary_size``.

    Parameters
    ----------
    sentence : sequence of str
        Tokenised, padded sentence (e.g. as produced by
        :func:`flatten_sentence_processing`).
    n_gram_counts : collections.Counter
        Counts of full ``n``-grams over the training corpus.
    vocabulary_size : int
        Vocabulary size ``V`` for the smoothing denominator.
    n : int, optional
        Order of the model. Defaults to ``3``.

    Returns
    -------
    list of tuple
        One ``(word, probability)`` per ``n``-gram in
        ``sentence``; ``word`` is the last token of the
        ``n``-gram (the predicted word).
    """
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
        if word not in [',', '.', '!', '?',':']:
            flat_words.append(word)

# Generate n-grams
n_grams = list(ngrams(flat_words, num_grams))
n_grams_1 = list(ngrams(flat_words, num_grams-1))

# Count the occurrences of each n-gram
n_gram_counts = Counter(n_grams)
n_gram_counts_1 = Counter(n_grams_1)


# Remove the 5 most common n-grams
most_common_n_grams = n_gram_counts.most_common(5)
for n_gram, _ in most_common_n_grams:
    del n_gram_counts[n_gram]

most_common_n_grams_1 = n_gram_counts_1.most_common(5)
for n_gram_1, _ in most_common_n_grams_1:
    del n_gram_counts_1[n_gram_1]
    

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
    """Tokenise, pad, and strip punctuation from ``sentence``.

    Runs ``nltk.word_tokenize(sentence)``, wraps the resulting
    token list with ``<s>`` left-padding and ``</s>``
    right-padding (using the module-level ``num_grams`` for the
    pad width), then drops the punctuation tokens
    ``,``, ``.``, ``!``, ``?``, ``:``. The stripped, padded list
    is what gets fed to :func:`calculate_word_probabilities`.

    Parameters
    ----------
    sentence : str
        Raw input sentence (lemmatised or surface form).

    Returns
    -------
    list of str
        Tokens after pad-and-filter, ready for n-gram scoring.
    """
    # Pad the sentence
    padded_sentence = list(pad_sequence(nltk.word_tokenize(sentence), pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=num_grams))
    flatten_sentence = []
    for word in padded_sentence:
        if word not in [',', '.', '!', '?',':']:
            flatten_sentence.append(word)
    return flatten_sentence

# test model
sentence_list = []
word_list = []
surprisal_list = []
probabilities_list = []
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
            probabilities_list.append(probability)
            word_list.append(word_sentence[j].lower())
            surprisal_list.append(-math.log2(probability))
            j+=1

df = pd.DataFrame({'Sentence': sentence_list, 'Word': word_list, 'Surprisal ngram-3': surprisal_list})
# Save the DataFrame to a CSV file 
define_name = f'word_surprisal_ngram{num_grams}_alpha{alpha}.csv'
csv_file_path = os.path.join('..', '..', 'podaci', define_name)
df.to_csv(csv_file_path, index=False)
