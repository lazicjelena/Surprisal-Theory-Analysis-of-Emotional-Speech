# -*- coding: utf-8 -*-
"""information_and_distance_functions.py

Created on Sat Nov 23 12:45:12 2024

@author: Jelena

Pipeline role
-------------
Library of information-theoretic and embedding-distance
helpers used by the embedding-based information measurement
pipeline. Provides cosine-similarity based context and
non-context embedding distances against a vocabulary sample,
sub-word to word reduction utilities (using the GPT-2
``space``-prefixed BPE marker ``Ġ``), and the per-sentence
information value / adjusted surprisal calculators that are
called by the embedding dataset builders upstream of
``build_dataset_for_embeddings.py``.
"""
import torch.nn.functional as F
import torch
import math

''' Funkcije'''

def non_context_embedding(word, vord, model, tokenizer, j):
  """Cosine-similarity-based non-contextual word embedding distance.

  Both ``word`` and ``vord`` are encoded independently with
  ``tokenizer`` and passed through ``model`` to retrieve the
  ``j``-th-from-last hidden state; the per-sub-word
  embeddings are mean-pooled into one vector per word. The
  returned similarity is the cosine similarity between the
  two pooled embeddings, rescaled from ``[-1, 1]`` to
  ``[0, 1]`` via ``0.5 * (sim + 1)``.

  Parameters
  ----------
  word, vord : str
      Words to compare in isolation.
  model : transformers.PreTrainedModel
      Encoder-style transformer model returning
      ``hidden_states``.
  tokenizer : transformers.PreTrainedTokenizer
      Matching tokenizer.
  j : int
      Reverse-index of the hidden state to use (``1`` is the
      last layer).

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
    """Reduce sub-word embeddings to one mean-pooled embedding per word.

    Walks the parallel ``subwords`` and ``subword_embeddings``
    lists and groups consecutive sub-words into words using the
    GPT-2 BPE convention (sub-words that begin a new word are
    prefixed by the special token ``Ġ``). The leading ``Ġ`` is
    stripped from the first sub-word of each word, and the
    embeddings of all sub-words belonging to the same word are
    mean-pooled into a single vector.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens produced by the tokenizer.
    subword_embeddings : sequence
        One embedding tensor per sub-token, in the same order.

    Returns
    -------
    words : list of str
        Reconstructed word sequence.
    word_embedding : list
        Mean-pooled embeddings, one per word.
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
    """Reduce sub-word probabilities to one product-pooled probability per word.

    Walks the parallel ``subwords`` and ``subword_probabilities``
    lists and groups consecutive sub-words into words using the
    GPT-2 BPE convention (sub-words that begin a new word are
    prefixed by the special token ``Ġ``). Within each word the
    sub-word probabilities are multiplied together so the
    returned per-word probability is the joint probability of
    the sub-word sequence.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens produced by the tokenizer.
    subword_probabilities : sequence
        One probability per sub-token, in the same order.

    Returns
    -------
    words : list of str
        Reconstructed word sequence.
    word_probabilities : list of float
        Joint sub-word probabilities, one per word.
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

def calculate_word_information_values(sentence, vocabulary_df, model, tokenizer):
    """Per-word information value: vocabulary-weighted embedding distance.

    For every word in ``sentence`` and every layer index
    ``j = 1..12`` the routine samples 50 vocabulary words from
    ``vocabulary_df`` (with a fixed random state for
    reproducibility), substitutes each vocabulary word into
    the sentence at the target position and computes:

    * the contextual cosine distance between the original
      word's embedding and the substituted word's embedding
      (rescaled to ``[0, 1]``),
    * the non-contextual cosine distance via
      :func:`non_context_embedding`,
    * the contextual probability of the substituted word
      retrieved through the model's softmax output.

    The returned per-word information value is the
    probability-weighted average distance: ``sum_v P(v|context)
    * dist(word, v)``, separately for the contextual (``ce_iv``)
    and non-contextual (``nce_iv``) distance.

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised sentence.
    vocabulary_df : pandas.DataFrame
        Vocabulary table with at least a ``word`` column.
    model : transformers.PreTrainedModel
        Model exposing both ``logits`` and ``hidden_states``.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer.

    Returns
    -------
    words_list : list of str
        Words of the input sentence.
    ce_iv : list
        Per-layer per-word contextual information values.
    nce_iv : list
        Per-layer per-word non-contextual information values.
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

def calculate_word_adjusted_surprisal(sentence, vocabulary_df, model, tokenizer):
    """Per-word adjusted surprisal: vocabulary-weighted surprisal of the substitution.

    For every word in ``sentence`` and every layer index
    ``j = 1..12`` the routine samples 50 vocabulary words from
    ``vocabulary_df`` (with a fixed random state for
    reproducibility), substitutes each vocabulary word into the
    sentence at the target position and accumulates a
    probability-weighted average similarity:

    * contextual: rescaled cosine similarity between the
      original and substituted word's contextual embedding,
    * non-contextual: rescaled cosine similarity from
      :func:`non_context_embedding`.

    Each accumulator is then converted to a base-2 surprisal
    via ``- log2(value + 1e-40)`` so the returned per-word
    quantity is interpretable as an adjusted surprisal.

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised sentence.
    vocabulary_df : pandas.DataFrame
        Vocabulary table with at least a ``word`` column.
    model : transformers.PreTrainedModel
        Model exposing both ``logits`` and ``hidden_states``.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer.

    Returns
    -------
    words_list : list of str
        Words of the input sentence.
    ce_iv : list
        Per-layer per-word contextual adjusted surprisals.
    nce_iv : list
        Per-layer per-word non-contextual adjusted surprisals.
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

          ce_iv[j][i] += 0.5 * (F.cosine_similarity(embedding_word1, embedding_word2, dim=0).item() + 1) * context_probability
          nce_iv[j][i] += non_context_embedding(word, vord, model, tokenizer, j) * context_probability

        ce_iv[j][i] = - math.log(ce_iv[j][i] + 1e-40, 2)
        nce_iv[j][i] = - math.log(nce_iv[j][i] + 1e-40, 2)

    return words_list, ce_iv, nce_iv