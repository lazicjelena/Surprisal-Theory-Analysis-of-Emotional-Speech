# -*- coding: utf-8 -*-
"""gpt2_contextual_entropy.py

Per-word contextual entropy estimation under GPT-2 (causal language
model). Converted from ``GPT_2_contextual_entropy.ipynb`` (P-013-NB).

Pipeline role
-------------
Information-theoretic alternative to per-word surprisal for the
``information_metrics/parameter_estimations/`` chain. For every
sentence in ``../../podaci/target_sentences.csv``, iterates over each
word slot, substitutes a fixed sample of vocabulary candidates from
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``, queries GPT-2 to
obtain per-subword softmax probabilities, aggregates sub-words back to
whole-word level and accumulates the binary-log entropy contribution
per slot. Sister of :mod:`yugo_gpt_contextual_entropy` (Serbian
YugoGPT variant). The resulting ``(Sentence, Word, Contextual
Entropy)`` table is written to
``../../podaci/contextual_entropy1.csv``.

Notes
-----
- Requires the ``transformers`` and ``torch`` packages.
- The model + tokenizer are loaded once at module import.
- The notebook samples ``n=5`` from the vocabulary inside
  ``calculate_contextual_entropy``; that body is preserved verbatim
  here.
"""

import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', '..', 'podaci', 'target_sentences.csv')
VOCABULARY_PATH = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_cleaned.csv')
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'contextual_entropy1.csv')

# Model configuration.
MODEL_NAME = 'gpt2'

# Module-level model / tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)


def extract_words_and_probabilities(subwords, subword_probabilities):
    """Merge GPT-2 BPE sub-tokens back into orthographic words.

    Walks ``subwords`` left to right; a token starting with the BPE
    word-boundary marker ``'Ġ'`` opens a new word, while every
    subsequent non-``Ġ`` token is concatenated to the current word
    and its probability is multiplied into the running word
    probability.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens as produced by the GPT-2 tokenizer.
    subword_probabilities : sequence of float or torch.Tensor
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


def calculate_contextual_entropy(sentence, tokenizer, model, vocab_df):
    """Per-word contextual entropy under GPT-2 for one sentence.

    For each word slot in ``sentence``, draws a fixed sample of
    vocabulary candidates (``vocabulary_df.sample(n=5,
    random_state=42)``), substitutes each candidate into the slot,
    encodes the modified sentence, queries the model to obtain
    per-token logits, averages the softmax over sequence positions
    to get one probability per sub-word, aggregates sub-words to
    whole-word level via :func:`extract_words_and_probabilities`,
    and accumulates the binary-log entropy contribution
    ``-p * log2(p + 1e-40)`` over candidates.

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised sentence whose words will be replaced
        slot by slot.
    tokenizer : transformers.PreTrainedTokenizer
        BPE tokenizer matching ``model``.
    model : transformers.PreTrainedModel
        Causal language model returning ``logits``.
    vocab_df : pandas.DataFrame
        Reference vocabulary with a ``word`` column (note: the
        function body re-samples the module-level
        ``vocabulary_df`` rather than ``vocab_df``; preserved
        verbatim from the source notebook).

    Returns
    -------
    words_list : list of str
        The original words from ``sentence``, in order.
    entropy_list : list of float
        Accumulated contextual entropy per slot, aligned with
        ``words_list``.
    """

    # calulate information value for one sentence
    words_list = []
    entropy_list = []

    # loop through all words in sentence
    for i in range(0, len(sentence.split(' '))):

      words = sentence.split(' ')
      words_list.append(words[i])

      entropy_list.append(0)
      vocab_df = vocabulary_df.sample(n=5, random_state=42).reset_index(drop=True)

      # loop through all vocabulary words
      for vord in vocab_df['word'].tolist():

        words[i] = vord
        # Tokenize the input sentence
        input_ids = tokenizer.encode(" ".join(words), return_tensors='pt')

        # Generate word probabilities using GPT-2 model
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Extract probabilities for each word
        subword_probabilities = torch.softmax(logits, dim=-1).mean(dim=1)

        # Decode the tokens back to words
        decoded_subwords = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        decoded_words2, probabilities = extract_words_and_probabilities(decoded_subwords, subword_probabilities[0])

        context_probability = probabilities[decoded_words2==vord].item()
        entropy_list[i] -= context_probability * math.log(context_probability + 1e-40, 2)

    return words_list, entropy_list


def main():
    """Build and save the per-word GPT-2 contextual-entropy table.

    Reads :data:`TARGET_SENTENCES_PATH` and :data:`VOCABULARY_PATH`,
    iterates target sentences from index ``14`` onwards (preserved
    verbatim from the source notebook), computes per-word entropy
    via :func:`calculate_contextual_entropy`, and writes the
    incremental ``(Sentence, Word, Contextual Entropy)`` table to
    :data:`OUTPUT_CSV_PATH` after each sentence.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)
    vocabulary_df = pd.read_csv(VOCABULARY_PATH)

    words_list = []
    target_sentence_list = []
    entropy_list = []

    # Save the DataFrame to a CSV file
    csv_file_path = OUTPUT_CSV_PATH

    for i in range(14,len(target_sentences_df)):
      sentence = target_sentences_df['Text'][i].lower()
      print(i)
      words, entropies = calculate_contextual_entropy(sentence.strip(), tokenizer, model, vocabulary_df)

      for word, entropy in zip(words, entropies):
        words_list.append(word)
        target_sentence_list.append(i)
        entropy_list.append(entropy)

      # Create a DataFrame
      df = pd.DataFrame({
          'Sentence': target_sentence_list,
          'Word': words_list,
          'Contextual Entropy': entropy_list
                         })
      df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
