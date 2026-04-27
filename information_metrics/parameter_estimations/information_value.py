# -*- coding: utf-8 -*-
"""information_value.py

Per-word *information value* under GPT-2: probability-weighted
distance between the original word and a sample of vocabulary
candidates, computed under four different similarity functions.
Converted from ``Information Value.ipynb`` (P-013-NB).

Pipeline role
-------------
Information-theoretic alternative to per-word surprisal for the
``information_metrics/parameter_estimations/`` chain. For every
sentence in ``../../podaci/target_sentences.csv``, iterates over each
word slot, substitutes a 50-row ``random_state=42`` sample of
vocabulary candidates from
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``, queries GPT-2
to obtain per-subword softmax probabilities and last-layer hidden
states, and accumulates four context-probability-weighted distances
per slot:

- ``Context Embedding`` -- contextual embedding cosine distance.
- ``Non-context Embedding`` -- non-contextual embedding cosine distance
  (separate forward passes per word).
- ``Pos-Tag`` -- POS-tag-mismatch (Serbian Stanza pipeline).
- ``Orthographic`` -- Levenshtein-based orthographic distance.

The resulting table is written incrementally to
``../../podaci/information_value.csv``.

Notes
-----
- Requires the ``transformers``, ``torch`` and ``stanza`` packages.
  The Serbian Stanza model is downloaded on first import (originally
  via ``stanza.download('sr')`` in the source notebook; the download
  call was removed in this conversion, run it once manually if the
  model is not yet on disk).
- Module-level model / tokenizer / NLP pipeline are loaded eagerly
  so the default arguments of the helpers resolve correctly.
"""

import math
import os
from difflib import SequenceMatcher

import pandas as pd
import stanza
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', '..', 'podaci', 'target_sentences.csv')
VOCABULARY_PATH = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_cleaned.csv')
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'information_value.csv')

# Model configuration.
MODEL_NAME = 'gpt2'

# Module-level model / tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Initialize the Stanza pipeline for Serbian
nlp = stanza.Pipeline('sr')


def non_context_embedding(word, vord, model, tokenizer):
    """Cosine similarity of *non-contextual* GPT-2 embeddings.

    Encodes ``word`` and ``vord`` independently, takes the
    sequence-mean of the model's hidden output for each, computes
    cosine similarity and rescales it from ``[-1, 1]`` to
    ``[0, 1]`` via ``0.5 * (cos + 1)``. "Non-contextual" because
    each word is run through the model on its own, without the
    surrounding sentence.

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

    Returns
    -------
    float
        Rescaled cosine similarity in ``[0, 1]``.
    """

    # Tokenize and get embedding for the target word
    word_input_ids = tokenizer.encode(word, return_tensors='pt')

    with torch.no_grad():
      output = model(word_input_ids)
      # Access hidden_states and get the last layer
      word_embedding = output[0].mean(dim=1)

    vocab_input_ids = tokenizer.encode(vord, return_tensors='pt')
    with torch.no_grad():
      output = model(vocab_input_ids)
      # Access hidden_states and get the last layer
      vocab_embedding = output[0].mean(dim=1)

    # Compute cosine similarity and normalize
    similarity = 0.5 * (F.cosine_similarity(word_embedding, vocab_embedding).item() + 1)

    return similarity


def get_pos_for_word_at_index(word, sentence, index):
    '''
    Return POS tag for a single word at index in sentence.
    '''
    # Process the sentence
    doc = nlp(sentence)

    # Loop through the tokens to find the POS tag at the specified index
    for sentence in doc.sentences:
        for token in sentence.tokens:
            if token.text == word and token.words[0].id == index+1:
                return token.words[0].upos  # Return the universal POS tag for the word at the given index

    return None  # Return None if the word at the specified index is not found


def pos_tags_similarity(word1, word2, sentence, ind):
    '''
    Calculate POS tag similarity between word1 and word2 in sentnece at ind.
    '''

    words = sentence.split(" ")

    words[ind] = word1
    p1 = get_pos_for_word_at_index(word1, " ".join(words), ind)

    words[ind] = word2
    p2 = get_pos_for_word_at_index(word2, " ".join(words), ind)

    if p1 == p2:
      return 1
    else:
      return 0


def levenshtein_distance(str1, str2):
    """Classical Levenshtein edit distance between two strings.

    Uses dynamic programming with a ``(len(str1)+1) x (len(str2)+1)``
    matrix; insertions, deletions and substitutions all cost ``1``.

    Parameters
    ----------
    str1 : str
        First string.
    str2 : str
        Second string.

    Returns
    -------
    int
        Minimum number of single-character edits required to
        transform ``str1`` into ``str2``.
    """
    # Create a distance matrix
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Initialize the matrix
    for i in range(len(str1) + 1):
        dp[i][0] = i  # Deletion
    for j in range(len(str2) + 1):
        dp[0][j] = j  # Insertion

    # Calculate distances
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[len(str1)][len(str2)]


def orthographic_similarity(word1, word2):
    """Length-normalised Levenshtein similarity in ``[0, 1]``.

    Computes ``1 - lev(word1, word2) / max(len(word1), len(word2))``,
    so identical strings score ``1`` and strings sharing no
    characters score ``0``.

    Parameters
    ----------
    word1 : str
        First word.
    word2 : str
        Second word.

    Returns
    -------
    float
        Orthographic similarity in ``[0, 1]``.
    """

    word1 = str(word1)
    word2 = str(word2)

    # Calculate similarity ratio between the two words
    d = levenshtein_distance(word1, word2)
    similarity = 1 - d/ max(len(word1), len(word2))

    return similarity


def sequence_matcher(word1, word2):
    """``difflib.SequenceMatcher`` ratio between two strings.

    Wraps :class:`difflib.SequenceMatcher` to return the matching
    ratio (also in ``[0, 1]``). Defined in the source notebook but
    not used by :func:`calculate_word_information_values`.

    Parameters
    ----------
    word1 : str
        First word.
    word2 : str
        Second word.

    Returns
    -------
    float
        ``SequenceMatcher.ratio()`` similarity.
    """

    word1 = str(word1)
    word2 = str(word2)

    # Calculate similarity ratio between the two words
    d = SequenceMatcher(None, word1, word2).ratio()

    return d


def extract_words_and_embeddings(subwords, subword_embeddings):
    """Aggregate GPT-2 BPE sub-token embeddings into word embeddings.

    A token starting with the BPE word-boundary marker ``'Ġ'``
    opens a new word; subsequent non-``Ġ`` tokens are appended to
    the current word and their embeddings are averaged once the
    word closes.

    Parameters
    ----------
    subwords : list of str
        BPE sub-tokens as produced by the GPT-2 tokenizer.
    subword_embeddings : sequence of torch.Tensor
        Per-sub-token embedding vectors aligned with ``subwords``.

    Returns
    -------
    words : list of str
        Merged orthographic words, with ``'Ġ'`` prefixes stripped.
    word_embedding : list of torch.Tensor
        Mean of the per-sub-token embeddings for each merged word.
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

    Companion to :func:`extract_words_and_embeddings`; multiplies
    sub-token probabilities into a single per-word probability.

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


def calculate_word_information_values(sentence, vocabulary_df, similarity_function, model = model, tokenizer = tokenizer):
    """Per-word information value under four similarity functions.

    For each word slot in ``sentence``, draws a 50-row
    ``random_state=42`` sample of ``vocabulary_df``, substitutes
    each candidate into the slot, and computes four
    context-probability-weighted distances against the original
    word: contextual embedding, non-contextual embedding, POS-tag
    mismatch and orthographic distance. The accumulated
    expectations are returned.

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised input sentence (lower-cased upstream).
    vocabulary_df : pandas.DataFrame
        Reference vocabulary with a ``word`` column.
    similarity_function : callable
        Kept verbatim from the source notebook (the actual
        POS-tag similarity is hard-coded inside the loop via
        :func:`pos_tags_similarity`); the parameter is unused.
    model : transformers.GPT2LMHeadModel, optional
        Defaults to the module-level ``model``.
    tokenizer : transformers.GPT2Tokenizer, optional
        Defaults to the module-level ``tokenizer``.

    Returns
    -------
    words_list : list of str
        Original words from ``sentence``, in order.
    ce_iv : list of float
        Context-embedding information value per slot.
    nce_iv : list of float
        Non-context embedding information value per slot.
    pt_iv : list of float
        POS-tag information value per slot.
    od_iv : list of float
        Orthographic-distance information value per slot.
    """

    #print(sentence)

    # calulate information value for one sentence
    words_list = []

    ce_iv = []
    nce_iv = []
    pt_iv = []
    od_iv = []

    words = sentence.split(' ')
    input_ids = tokenizer.encode(" ".join(words), return_tensors='pt')

    # Forward pass to get hidden states
    with torch.no_grad():
      outputs = model(input_ids, output_hidden_states=True)
      last_hidden_state = outputs.hidden_states[-1]  # This is the final layer's hidden state for each token

    decoded_subwords = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    decoded_words1, embeddings1 = extract_words_and_embeddings(decoded_subwords, last_hidden_state[0,:])

    # loop through all words in sentence
    for i in range(0, len(sentence.split(' '))):

      words = sentence.split(' ')
      word = words[i]
      #print(word)
      words_list.append(words[i])

      ce_iv.append(0)
      nce_iv.append(0)
      pt_iv.append(0)
      od_iv.append(0)

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
            last_hidden_state = outputs.hidden_states[-1]  # This is the final layer's hidden state for each token

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
        non_contextual_embedding_distance = 1 - non_context_embedding(word, vord, model, tokenizer)
        pos_tag_distance = 1 - pos_tags_similarity(word, vord, sentence, i)
        orthographic_distance = 1 - orthographic_similarity(word, vord)

        ce_iv[i] += contextual_embedding_distance * context_probability
        nce_iv[i] += non_contextual_embedding_distance * context_probability
        pt_iv[i] += pos_tag_distance * context_probability
        od_iv[i] += orthographic_distance * context_probability

    return words_list, ce_iv, nce_iv, pt_iv, od_iv


def main():
    """Build and save the per-word four-distance information-value table.

    Reads :data:`TARGET_SENTENCES_PATH` and :data:`VOCABULARY_PATH`,
    iterates target sentences from index ``57`` onwards (preserved
    verbatim from the source notebook), computes per-word
    information values via :func:`calculate_word_information_values`
    using :func:`pos_tags_similarity` as the similarity hook, and
    writes the incremental ``(Sentence, Word, Context Embedding,
    Non-context Embedding, Pos-Tag, Orthographic)`` table to
    :data:`OUTPUT_CSV_PATH` after each sentence.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)
    vocabulary_df = pd.read_csv(VOCABULARY_PATH)

    words_list = []
    target_sentence_list = []
    ce_iv_list = []
    nce_iv_list = []
    pt_iv_list = []
    od_iv_list = []

    # Save the DataFrame to a CSV file
    csv_file_path = OUTPUT_CSV_PATH

    for i in range(57,len(target_sentences_df)):
      sentence = target_sentences_df['Text'][i].lower()
      print(i)
      words, ce_ivs, nce_ivs, pt_ivs, od_ivs = calculate_word_information_values(sentence.strip(), vocabulary_df, pos_tags_similarity)

      for word, ce_iv, nce_iv, pt_iv, od_iv in zip(words, ce_ivs, nce_ivs, pt_ivs, od_ivs):
        words_list.append(word)
        target_sentence_list.append(i)
        ce_iv_list.append(ce_iv)
        nce_iv_list.append(nce_iv)
        pt_iv_list.append(pt_iv)
        od_iv_list.append(od_iv)

      # Create a DataFrame
      df = pd.DataFrame({
          'Sentence': target_sentence_list,
          'Word': words_list,
          'Context Embedding': ce_iv_list,
          'Non-context Embedding': nce_iv_list,
          'Pos-Tag': pt_iv_list,
          'Orthographic': od_iv_list
                         })
      df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
