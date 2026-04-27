# -*- coding: utf-8 -*-
"""information_value_embeddings_bertic.py

Per-word *information value* across all 12 BERTic hidden layers, under
contextual embedding similarity. Converted from
``Information Value Embeddings BERTic.ipynb`` (P-013-NB).

Pipeline role
-------------
BERTic (``classla/bcms-bertic`` masked-LM trained on Bosnian /
Croatian / Montenegrin / Serbian) variant of
:mod:`information_value_embeddings`. For every sentence in
``../../podaci/target_sentences.csv``, iterates over each word slot,
substitutes a 50-row ``random_state=42`` sample of vocabulary
candidates from
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``, queries the
model with ``output_hidden_states=True``, and for each hidden-layer
index ``j in 1..12`` accumulates a context-probability-weighted
contextual-embedding distance per slot (column ``CE j``). The result
is written incrementally to
``../../podaci/information_value_1.csv``.

The source notebook concluded with two extra cells that concatenated
``information_value_0.csv`` and ``information_value_1.csv`` into
``information_value_bertic.csv``; that merge step is intentionally
omitted here -- run it manually in pandas if both halves are
available.

Notes
-----
- Requires the ``transformers`` and ``torch`` packages.
- Module-level model / tokenizer are loaded eagerly.
"""

import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizer

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', '..', 'podaci', 'target_sentences.csv')
VOCABULARY_PATH = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_cleaned.csv')
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'information_value_1.csv')

# Model configuration.
MODEL_NAME = "classla/bcms-bertic"

# Module-level model / tokenizer.
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)


def mask_each_word(sentence):
    """Produce one masked-sentence variant per BERT sub-token.

    Tokenizes ``sentence`` with the module-level tokenizer and,
    for each tokenized position, returns a copy of the sentence
    with that sub-token replaced by ``[MASK]``. The original
    sub-token is also returned so the caller can score its
    probability under the masked LM.

    Parameters
    ----------
    sentence : str
        Raw input sentence (lower-cased upstream).

    Returns
    -------
    masked_sentences : list of str
        One sentence per sub-token position with ``[MASK]``
        inserted.
    masked_words : list of str
        The original sub-tokens that were replaced.
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


def estimate_masked_probability(sentence, candidate_word, model = model, tokenizer = tokenizer):
    """Probability of ``candidate_word`` at the ``[MASK]`` position.

    Parameters
    ----------
    sentence : str
        Sentence containing one ``[MASK]`` token.
    candidate_word : str
        The sub-token whose conditional probability is scored.
    model : transformers.BertForMaskedLM, optional
        Defaults to the module-level ``model``.
    tokenizer : transformers.BertTokenizer, optional
        Defaults to the module-level ``tokenizer``.

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


def extract_words_and_probabilities(words, probabilities):
    """Merge BERT WordPiece sub-tokens back into orthographic words.

    Glues each ``##``-prefixed continuation to the preceding head
    and multiplies the matching per-piece probabilities.

    Parameters
    ----------
    words : list of str
        WordPiece tokens.
    probabilities : list of float
        Per-piece probabilities aligned with ``words``.

    Returns
    -------
    new_words : list of str
        Merged orthographic words, with ``##`` prefixes stripped.
    new_probabilities : list of float
        Product of per-piece probabilities for each merged word.
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


def extract_words_and_embeddings(subwords, subword_embeddings):
    """Aggregate WordPiece sub-token embeddings into word embeddings.

    A token starting with ``##`` is treated as a continuation of
    the previous word; its embedding is averaged into the word's
    final embedding.

    Parameters
    ----------
    subwords : list of str
        WordPiece tokens.
    subword_embeddings : sequence of torch.Tensor
        Per-sub-token embeddings aligned with ``subwords``.

    Returns
    -------
    words : list of str
        Merged orthographic words.
    word_embeddings : list of torch.Tensor
        Mean of per-sub-token embeddings for each merged word.
    """
    words = []
    word_embeddings = []

    current_word = ""
    current_subword_embeddings = []

    for subword, embedding in zip(subwords, subword_embeddings):
        # Check if the subword is a continuation (starts with '##')
        if subword.startswith('##'):
            # Remove '##' and concatenate
            current_word += subword[2:]
            current_subword_embeddings.append(embedding)
        else:
            # If we have a current word, save it before starting new one
            if current_word:
                words.append(current_word)
                # Average all subword embeddings for the word
                word_embeddings.append(
                    sum(current_subword_embeddings) / len(current_subword_embeddings)
                )

            # Start new word
            current_word = subword
            current_subword_embeddings = [embedding]

    # Add the last word if exists
    if current_word:
        words.append(current_word)
        word_embeddings.append(
            sum(current_subword_embeddings) / len(current_subword_embeddings)
        )

    return words, word_embeddings


def calculate_word_information_values(sentence, vocabulary_df, model = model, tokenizer = tokenizer):
    """Per-word, per-layer contextual-embedding information value.

    For each word slot in ``sentence`` and each hidden-layer index
    ``j in 1..12``, draws a 50-row ``random_state=42`` sample of
    ``vocabulary_df``, substitutes each candidate into the slot,
    and accumulates a context-probability-weighted contextual
    embedding distance against the original word.

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised input sentence (lower-cased upstream).
    vocabulary_df : pandas.DataFrame
        Reference vocabulary with a ``word`` column.
    model : transformers.BertForMaskedLM, optional
        Defaults to the module-level ``model``.
    tokenizer : transformers.BertTokenizer, optional
        Defaults to the module-level ``tokenizer``.

    Returns
    -------
    words_list : list of str
        Original words from ``sentence``, in order.
    ce_iv : list of list of float
        Contextual-embedding information value per layer
        (index ``1..12``); ``ce_iv[0]`` is unused.
    """

    # calulate information value for one sentence
    words_list = sentence.split(' ')

    ce_iv = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

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
          ce_iv[j][i] += contextual_embedding_distance * context_probability

    return words_list, ce_iv


def main():
    """Build and save the per-word, per-layer information-value table.

    Reads :data:`TARGET_SENTENCES_PATH` and :data:`VOCABULARY_PATH`,
    iterates target sentences from index ``59`` onwards (preserved
    verbatim from the source notebook), computes per-word per-layer
    information values via :func:`calculate_word_information_values`,
    and writes the incremental ``(Sentence, Word, CE 1..12)`` table
    to :data:`OUTPUT_CSV_PATH` after each sentence.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)
    vocabulary_df = pd.read_csv(VOCABULARY_PATH)

    words_list = []
    target_sentence_list = []
    ce_iv_list = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

    # Save the DataFrame to a CSV file
    csv_file_path = OUTPUT_CSV_PATH

    for i in range(59,len(target_sentences_df)):
      sentence = target_sentences_df['Text'][i].lower()
      print(i)
      print(sentence)
      words, ce_ivs = calculate_word_information_values(sentence.strip(), vocabulary_df)

      for ind in range(0,len(words)):
        words_list.append(words[ind])
        target_sentence_list.append(i)
        for j in range(1,13):
          ce_iv_list[j].append(ce_ivs[j][ind])

      # Create a DataFrame
      df = pd.DataFrame({
          'Sentence': target_sentence_list,
          'Word': words_list,
          **{f'CE {j}': ce_iv_list[j] for j in range(1, 13)}}
          )
      df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
