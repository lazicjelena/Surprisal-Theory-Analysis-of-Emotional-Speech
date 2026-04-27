# -*- coding: utf-8 -*-
"""bert_surprisal_estimation.py

Per-word surprisal estimation using multilingual BERT (masked language
model). Converted from ``BERT surprisal estimation.ipynb`` (P-013-NB).

Pipeline role
-------------
LLM-based per-word surprisal estimator for the
``surprisal_estimation/`` chain. Iterates over every sentence in
``../podaci/target_sentences.csv``, masks each token in turn, asks the
``bert-base-multilingual-uncased`` model for the probability of the
original token at the masked position, merges WordPiece sub-tokens back
into orthographic words and writes the per-word surprisal column
``Surprisal BERT`` to ``../podaci/word_surprisals_bert.csv``. The
output is a sibling of ``word_surprisals_yugo.csv`` produced by
``yugo_gpt3_surprisal_estimation.py`` and is consumed by the
``information_metrics/`` build_dataset stage.

Notes
-----
- Requires the ``transformers`` and ``torch`` packages.
- The model + tokenizer are loaded once at module import, then
  used as default arguments inside helpers.
"""

import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizer

# Paths (relative to project root, matching yugo_gpt3_surprisal_estimation.py
# convention).
TARGET_SENTENCES_PATH = os.path.join('..', 'podaci', 'target_sentences.csv')
OUTPUT_CSV_PATH = os.path.join('..', 'podaci', 'word_surprisals_bert.csv')

# Model configuration.
MODEL_NAME = 'bert-base-multilingual-uncased'

# Module-level model / tokenizer (loaded eagerly so that the default
# arguments of estimate_masked_probability resolve correctly).
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)


def mask_each_word(sentence):
    """Produce one masked-sentence variant per token in ``sentence``.

    Tokenizes ``sentence`` with the module-level BERT tokenizer
    and, for each tokenized position, returns a copy of the
    sentence with that token replaced by ``[MASK]``. The original
    masked token is also returned alongside, so that the caller
    can later score its probability under the masked LM.

    Parameters
    ----------
    sentence : str
        Raw input sentence (lower-cased upstream).

    Returns
    -------
    masked_sentences : list of str
        Length equals the tokenized sentence length; element ``i``
        is the sentence with token ``i`` replaced by ``[MASK]``.
    masked_words : list of str
        The original sub-tokens that were replaced, in the same
        order as ``masked_sentences``.
    """
    # Tokenize the sentence
    tokenized_sentence = tokenizer.tokenize(sentence)

    # Lists to store masked sentences and masked words
    masked_sentences = []
    masked_words = []

    # Iterate through each word and replace it with [MASK]
    for i in range(len(tokenized_sentence)):
        masked_sentence = list(tokenized_sentence)  # Create a copy of the tokenized sentence
        masked_sentence[i] = tokenizer.mask_token  # Replace the i-th word with [MASK]

        # Add the masked sentence to the list
        masked_sentences.append(tokenizer.convert_tokens_to_string(masked_sentence))

        # Add the masked word to the list
        masked_words.append(tokenized_sentence[i])

    return masked_sentences, masked_words


def estimate_masked_probability(sentence, candidate_word, model=model, tokenizer=tokenizer):
    """Probability of ``candidate_word`` at the ``[MASK]`` position.

    Encodes ``sentence`` (which must contain exactly one ``[MASK]``
    token) into input ids, replaces the mask id with the id of
    ``candidate_word``, runs the masked LM forward pass, applies
    softmax over the vocabulary at the masked position, and
    returns the probability assigned to ``candidate_word``.

    Parameters
    ----------
    sentence : str
        Sentence containing one ``[MASK]`` token.
    candidate_word : str
        The sub-token whose conditional probability is to be
        scored.
    model : transformers.BertForMaskedLM, optional
        BERT MLM checkpoint. Defaults to the module-level
        ``model``.
    tokenizer : transformers.BertTokenizer, optional
        Matching tokenizer. Defaults to the module-level
        ``tokenizer``.

    Returns
    -------
    float
        Softmax probability of ``candidate_word`` at the masked
        position; in ``[0, 1]``.
    """
    # Tokenize the input sentence
    tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

    # Find the index of the [MASK] token
    mask_index = tokenized_sentence.index(tokenizer.mask_token_id)

    # Replace [MASK] with the candidate word
    tokenized_sentence[mask_index] = tokenizer.convert_tokens_to_ids(candidate_word)

    # Convert tokenized sequence to PyTorch tensor
    input_ids = torch.tensor([tokenized_sentence])

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Apply softmax to obtain probabilities
    probabilities = F.softmax(logits[0, mask_index], dim=0)

    # Find the probability of the candidate word
    candidate_index = tokenizer.convert_tokens_to_ids(candidate_word)
    candidate_probability = probabilities[candidate_index].item()

    return candidate_probability


def concatenate_words_and_probabilities(words, probabilities):
    """Merge BERT WordPiece sub-tokens back into orthographic words.

    BERT splits unknown tokens into ``"head"`` plus one or more
    ``"##suffix"`` continuations. This routine glues each
    ``##``-prefixed continuation to the preceding head and
    multiplies the matching per-piece probabilities to obtain a
    single probability for the merged orthographic word.

    Parameters
    ----------
    words : list of str
        WordPiece tokens as produced by the BERT tokenizer.
    probabilities : list of float
        Per-piece probabilities aligned with ``words``.

    Returns
    -------
    new_words : list of str
        Merged orthographic words, with ``##`` prefixes stripped.
    new_probabilities : list of float
        Product of the per-piece probabilities for each merged
        word; length matches ``new_words``.
    """
    new_words = []
    new_probabilities = []

    i = 0
    while i < len(words):
        current_word = words[i]
        current_probability = probabilities[i]

        while i + 1 < len(words) and words[i + 1].startswith('##'):
            next_word = words[i + 1][2:]
            current_word += next_word
            current_probability *= probabilities[i + 1]
            i += 1  # Move to the next word in the sequence

        new_words.append(current_word)
        new_probabilities.append(current_probability)

        i += 1

    return new_words, new_probabilities


def calculate_word_probabilities(sentence):
    """Per-word probabilities under the masked LM for one sentence.

    Pipeline: :func:`mask_each_word` produces one mask variant per
    sub-token; :func:`estimate_masked_probability` scores each
    masked token under the model;
    :func:`concatenate_words_and_probabilities` merges WordPiece
    fragments back to orthographic words. The product of the
    per-piece probabilities is also returned as a sentence-level
    likelihood proxy.

    Parameters
    ----------
    sentence : str
        Input sentence (lower-cased upstream).

    Returns
    -------
    words : list of str
        Orthographic words after WordPiece merging.
    probabilities : list of float
        Per-word probabilities aligned with ``words``.
    total : float
        Product of all per-piece probabilities (kept for parity
        with the original notebook output).
    """
    masked_sentences, masked_words = mask_each_word(sentence)

    list_probabilities = []
    total = 1
    for candidate, masked_sentence in zip(masked_words, masked_sentences):
        probability = estimate_masked_probability(masked_sentence, candidate)
        list_probabilities.append(probability)
        total = total * probability

    words, probabilities = concatenate_words_and_probabilities(masked_words, list_probabilities)

    return words, probabilities, total


def main():
    """Build and save the per-word BERT surprisal table.

    Reads :data:`TARGET_SENTENCES_PATH`, iterates each row's
    lower-cased sentence, computes per-word probabilities with
    :func:`calculate_word_probabilities`, converts to surprisal
    via ``-log2(p)``, and writes the resulting
    ``(Sentence, Word, Surprisal BERT)`` table to
    :data:`OUTPUT_CSV_PATH`.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)

    words_list = []
    probabilities_list = []
    target_sentence_list = []

    for i in range(0, len(target_sentences_df)):
        sentence = target_sentences_df['Text'][i].lower()
        words = sentence.split(' ')
        _, probabilities, total = calculate_word_probabilities(sentence)

        for word, prob in zip(words, probabilities):
            words_list.append(word)
            probabilities_list.append(-math.log2(prob))
            target_sentence_list.append(i)

    # Create a DataFrame
    df = pd.DataFrame({'Sentence': target_sentence_list, 'Word': words_list, 'Surprisal BERT': probabilities_list})

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
