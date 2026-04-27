# -*- coding: utf-8 -*-
"""information_value_embeddings.py

Per-word *information value* across all 12 GPT-2 hidden layers, under
both contextual and non-contextual embedding similarity. Converted
from ``Information Value Embeddings.ipynb`` (P-013-NB).

Pipeline role
-------------
Layer-resolved variant of :mod:`information_value` for the
``information_metrics/parameter_estimations/`` chain. For every
sentence in ``../../podaci/target_sentences.csv``, iterates over each
word slot, substitutes a 50-row ``random_state=42`` sample of
vocabulary candidates from
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``, queries GPT-2
with ``output_hidden_states=True``, and for each hidden-layer index
``j in 1..12`` accumulates two context-probability-weighted distances
per slot: contextual embedding (``CE j``) and non-contextual embedding
(``NCE j``). The result is written incrementally to
``../../podaci/information_value2.csv``.

Notes
-----
- Requires the ``transformers`` and ``torch`` packages.
- Module-level model / tokenizer are loaded eagerly so the default
  arguments of :func:`calculate_word_information_values` resolve.
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
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'information_value2.csv')

# Model configuration.
MODEL_NAME = 'gpt2'

# Module-level model / tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)


def non_context_embedding(word, vord, model, tokenizer, j):
    """Layer-``j`` non-contextual cosine similarity between two words.

    Encodes ``word`` and ``vord`` independently and reads the
    ``j``-th-from-last hidden state for each, takes the
    sequence-mean and returns the rescaled cosine
    ``0.5 * (cos + 1)``.

    Parameters
    ----------
    word : str
        Original target word.
    vord : str
        Candidate vocabulary word.
    model : transformers.GPT2LMHeadModel
        Causal-LM checkpoint.
    tokenizer : transformers.GPT2Tokenizer
        Matching tokenizer.
    j : int
        ``hidden_states[-j]`` selects the j-th layer from the top.

    Returns
    -------
    float
        Rescaled cosine similarity in ``[0, 1]``.
    """

    # Tokenize and get embedding for the target word
    word_input_ids = tokenizer.encode(word, return_tensors='pt')

    with torch.no_grad():
      outputs = model(word_input_ids, output_hidden_states=True)
      # Access hidden_states and get the last layer
      word_embedding = outputs.hidden_states[-j].mean(dim=1)

    vocab_input_ids = tokenizer.encode(vord, return_tensors='pt')
    with torch.no_grad():
      outputs = model(vocab_input_ids, output_hidden_states=True)
      # Access hidden_states and get the last layer
      vocab_embedding = outputs.hidden_states[-j].mean(dim=1)

    # Compute cosine similarity and normalize
    similarity = 0.5 * (F.cosine_similarity(word_embedding, vocab_embedding).item() + 1)

    return similarity


def extract_words_and_embeddings(subwords, subword_embeddings):
    """Aggregate GPT-2 BPE sub-token embeddings into word embeddings.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens.
    subword_embeddings : sequence of torch.Tensor
        Per-sub-token embedding vectors.

    Returns
    -------
    words : list of str
        Merged orthographic words.
    word_embedding : list of torch.Tensor
        Mean of per-sub-token embeddings for each merged word.
    """
    words = []
    word_embedding = []

    current_word = ""
    current_embedding = []

    for subword, embedding in zip(subwords, subword_embeddings):
        # Check if the subword starts with the special token 'Ġ'
        if subword.startswith('Ġ'):
            if current_word:
                words.append(current_word)
                word_embedding.append(sum(current_embedding) / len(current_embedding))

            # Reset current word and probability for the new word
            current_word = subword[1:]  # Remove 'Ġ' from the start
            current_embedding= [embedding]
        else:
            # Concatenate subwords to form the current word
            current_word += subword
            # Multiply probabilities for subwords within the same word
            current_embedding.append(embedding)

    # Add the last word and its probability
    if current_word:
        words.append(current_word)
        word_embedding.append(sum(current_embedding) / len(current_embedding))

    return words, word_embedding


def extract_words_and_probabilities(subwords, subword_probabilities):
    """Merge GPT-2 BPE sub-tokens back into orthographic words.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens.
    subword_probabilities : sequence of float or torch.Tensor
        Per-piece probabilities aligned with ``subwords``.

    Returns
    -------
    words : list of str
        Merged orthographic words.
    word_probabilities : list of float or torch.Tensor
        Product of per-piece probabilities for each merged word.
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


def calculate_word_information_values(sentence, vocabulary_df, model = model, tokenizer = tokenizer):
    """Per-word, per-layer information value for one sentence.

    For each word slot in ``sentence`` and each hidden-layer
    index ``j in 1..12``, draws a 50-row ``random_state=42``
    sample of ``vocabulary_df``, substitutes each candidate into
    the slot, and accumulates two context-probability-weighted
    distances against the original word: contextual embedding
    (cosine distance at layer ``-j``) and non-contextual
    embedding (via :func:`non_context_embedding`).

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised input sentence (lower-cased upstream).
    vocabulary_df : pandas.DataFrame
        Reference vocabulary with a ``word`` column.
    model : transformers.GPT2LMHeadModel, optional
        Defaults to the module-level ``model``.
    tokenizer : transformers.GPT2Tokenizer, optional
        Defaults to the module-level ``tokenizer``.

    Returns
    -------
    words_list : list of str
        Original words from ``sentence``, in order.
    ce_iv : list of list of float
        Contextual-embedding information value per layer (index
        ``1..12``); ``ce_iv[0]`` is unused / empty.
    nce_iv : list of list of float
        Non-contextual-embedding information value per layer.
    """

    # calulate information value for one sentence
    words_list = sentence.split(' ')

    ce_iv = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    nce_iv = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

    words = sentence.split(' ')
    input_ids = tokenizer.encode(" ".join(words), return_tensors='pt')

    # Forward pass to get hidden states
    with torch.no_grad():
      outputs = model(input_ids, output_hidden_states=True)

    decoded_subwords = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    for j in range(1,13):
      last_hidden_state = outputs.hidden_states[-j]
      decoded_words1, embeddings1 = extract_words_and_embeddings(decoded_subwords, last_hidden_state[0,:])

      # loop through all words in sentence
      for i in range(0, len(sentence.split(' '))):

        words = sentence.split(' ')
        word = words[i]
        ce_iv[j].append(0)
        nce_iv[j].append(0)

        embedding_word1 = embeddings1[i]

        vocab_df = vocabulary_df.sample(n=50, random_state=42).reset_index(drop=True)
        # loop through all vocabulary words
        for vord in vocab_df['word'].tolist():

          words[i] = vord
          # Tokenize the input sentence
          input_ids = tokenizer.encode(" ".join(words), return_tensors='pt')

          # Generate word probabilities using GPT-2 model
          with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-j]  # This is the final layer's hidden state for each token

          # Extract probabilities for each word
          subword_probabilities = torch.softmax(logits, dim=-1).mean(dim=2)

          # Decode the tokens back to words
          decoded_subwords = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
          decoded_words2, probabilities = extract_words_and_probabilities(decoded_subwords, subword_probabilities[0])

          decoded_words2, embeddings2 = extract_words_and_embeddings(decoded_subwords, last_hidden_state[0,:])
          embedding_word2 = embeddings2[i]

          context_probability = probabilities[decoded_words2==vord].item()

          # distances
          contextual_embedding_distance = 1 - 0.5 * (F.cosine_similarity(embedding_word1, embedding_word2, dim=0).item() + 1)
          non_contextual_embedding_distance = 1 - non_context_embedding(word, vord, model, tokenizer, j)

          ce_iv[j][i] += contextual_embedding_distance * context_probability
          nce_iv[j][i] += non_contextual_embedding_distance * context_probability

    return words_list, ce_iv, nce_iv


def main():
    """Build and save the per-word, per-layer information-value table.

    Reads :data:`TARGET_SENTENCES_PATH` and :data:`VOCABULARY_PATH`,
    iterates target sentences over the range ``47..50`` (preserved
    verbatim from the source notebook), computes per-word per-layer
    information values via :func:`calculate_word_information_values`,
    and writes the incremental ``(Sentence, Word, CE 1..12, NCE 1..12)``
    table to :data:`OUTPUT_CSV_PATH` after each sentence.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)
    vocabulary_df = pd.read_csv(VOCABULARY_PATH)

    words_list = []
    target_sentence_list = []
    ce_iv_list = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    nce_iv_list = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

    # Save the DataFrame to a CSV file
    csv_file_path = OUTPUT_CSV_PATH

    for i in range(47,50):
      sentence = target_sentences_df['Text'][i].lower()
      print(i)
      words, ce_ivs, nce_ivs = calculate_word_information_values(sentence.strip(), vocabulary_df)

      for ind in range(0,len(words)):
        words_list.append(words[ind])
        target_sentence_list.append(i)
        for j in range(1,13):
          ce_iv_list[j].append(ce_ivs[j][ind])
          nce_iv_list[j].append(nce_ivs[j][ind])

      # Create a DataFrame
      df = pd.DataFrame({
          'Sentence': target_sentence_list,
          'Word': words_list,
          **{f'CE {j}': ce_iv_list[j] for j in range(1, 13)},
          **{f'NCE {j}': nce_iv_list[j] for j in range(1, 13)}
                         })
      df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
