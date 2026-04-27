# -*- coding: utf-8 -*-
"""gpt3_surprisal_estimation.py

Per-word surprisal estimation using GPT-Neo 2.7B (the open
re-implementation of GPT-3). Converted from
``GPT-3 surprisal estimation.ipynb`` (P-013-NB).

Pipeline role
-------------
LLM-based per-word surprisal estimator for the
``surprisal_estimation/`` chain. Uses the
``EleutherAI/gpt-neo-2.7B`` checkpoint as a stand-in for OpenAI's
GPT-3. Iterates over every sentence in
``../podaci/target_sentences.csv``, encodes the full sentence, takes
the time-averaged softmax of the model logits at each sub-token
position, merges BPE sub-tokens back into orthographic words and
writes the per-word surprisal column ``Surprisal GPT-3`` to
``../podaci/word_surprisals_gpt3.csv``. Distinct from
:mod:`yugo_gpt3_surprisal_estimation`, which uses the South-Slavic
``YugoGPT`` checkpoint instead.

Notes
-----
- Requires the ``transformers`` and ``torch`` packages.
- The model + tokenizer are loaded once at module import.
"""

import math
import os

import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', 'podaci', 'target_sentences.csv')
OUTPUT_CSV_PATH = os.path.join('..', 'podaci', 'word_surprisals_gpt3.csv')

# Model configuration.
MODEL_NAME = 'EleutherAI/gpt-neo-2.7B'

# Module-level model / tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)


def extract_words_and_probabilities(subwords, subword_probabilities):
    """Merge GPT-Neo BPE sub-tokens back into orthographic words.

    GPT-Neo (like GPT-2) prefixes each new word with the special
    token ``'Ġ'`` (U+0120). This routine walks the sub-token /
    probability pairs and groups them into orthographic words: a
    sub-token that starts with ``'Ġ'`` opens a new word, while
    every subsequent non-``Ġ`` sub-token is concatenated to the
    current word and its probability is multiplied into the
    running word probability.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens as produced by the GPT-Neo tokenizer.
    subword_probabilities : list of float or torch.Tensor
        Per-piece probabilities aligned with ``subwords``.

    Returns
    -------
    words : list of str
        Merged orthographic words, with ``'Ġ'`` prefixes stripped.
    word_probabilities : list of float or torch.Tensor
        Product of the per-piece probabilities for each merged
        word; length matches ``words``.
    """
    words = []
    word_probabilities = []

    current_word = ""
    current_probability = 1.0  # Initialize to 1.0 as we will multiply probabilities

    # Iterate through the subword probabilities
    for subword, probability in zip(subwords, subword_probabilities):
        # Check if the subword starts with the special token 'Ġ'
        if subword.startswith('Ġ'):
            # If we have a current word, add it to the list with its probability
            if current_word:
                words.append(current_word)
                word_probabilities.append(current_probability)

            # Reset current word and probability for the new word
            current_word = subword[1:]  # Remove 'Ġ' from the start
            current_probability = probability
        else:
            # Concatenate subwords to form the current word
            current_word += subword
            # Multiply probabilities for subwords within the same word
            current_probability *= probability

    # Add the last word and its probability
    if current_word:
        words.append(current_word)
        word_probabilities.append(current_probability)

    return words, word_probabilities


def calculate_word_probabilities(sentence, tokenizer=tokenizer, model=model):
    """Per-word probabilities under GPT-Neo for one sentence.

    Encodes ``sentence`` with the BPE tokenizer, runs a single
    GPT-Neo forward pass, and approximates per-position
    probabilities by averaging the softmax over the time
    dimension (preserved verbatim from the source notebook).
    The decoded sub-tokens are then merged into orthographic
    words via :func:`extract_words_and_probabilities`.

    Parameters
    ----------
    sentence : str
        Input sentence (lower-cased upstream).
    tokenizer : transformers.GPT2Tokenizer, optional
        Defaults to the module-level ``tokenizer``.
    model : transformers.GPTNeoForCausalLM, optional
        Defaults to the module-level ``model``.

    Returns
    -------
    words : list of str
        Orthographic words after BPE merging.
    probabilities : list of torch.Tensor
        Per-word probabilities aligned with ``words``.
    total_probability : float
        Sum of per-piece probabilities (kept for parity with the
        original notebook output).
    """
    # Tokenize the input sentence
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    # Generate word probabilities using GPT-2 model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Extract probabilities for each word
    word_probabilities = torch.softmax(logits, dim=-1).mean(dim=1)

    # Decode the tokens back to words
    decoded_words = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    total_probability = 0
    subwords = []
    subwords_probabilities = []
    # Display word probabilities for each word
    for word, probability in zip(decoded_words, word_probabilities[0]):
        subwords.append(word)
        subwords_probabilities.append(probability)
        total_probability += probability

    words, probabilities = extract_words_and_probabilities(subwords, subwords_probabilities)

    return words, probabilities, total_probability


def main():
    """Build and save the per-word GPT-3 surprisal table.

    Reads :data:`TARGET_SENTENCES_PATH`, iterates each row's
    lower-cased sentence, computes per-word probabilities with
    :func:`calculate_word_probabilities`, converts to surprisal
    via ``-log2(p.item())``, and writes the resulting
    ``(Sentence, Word, Surprisal GPT-3)`` table to
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
            probabilities_list.append(-math.log2(prob.item()))
            target_sentence_list.append(i)

    # Create a DataFrame
    df = pd.DataFrame({'Sentence': target_sentence_list, 'Word': words_list, 'Surprisal GPT-3': probabilities_list})

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
