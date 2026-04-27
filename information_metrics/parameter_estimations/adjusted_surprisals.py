# -*- coding: utf-8 -*-
"""adjusted_surprisals.py

Per-word *adjusted surprisal* under GPT-2: probability-weighted
similarity between the original word and a sample of vocabulary
candidates, then ``-log2`` of the expected similarity. Computed under
four different similarity functions. Converted from
``Adjusted Surprisals.ipynb`` (P-013-NB).

Pipeline role
-------------
Sister of :mod:`information_value` -- same model, same vocabulary
sample, same four similarity hooks, but the per-slot quantity is
``-log2(E[similarity])`` instead of ``E[1 - similarity]``. For every
sentence in ``../../podaci/target_sentences.csv``, iterates over each
word slot, substitutes a 50-row ``random_state=42`` sample of
candidates from
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``, queries GPT-2
to obtain per-subword softmax probabilities and last-layer hidden
states, accumulates context-probability-weighted similarities under
contextual embedding, non-contextual embedding, POS-tag and
orthographic distance, and converts each accumulator to a binary
log-surprisal. The result is written incrementally to
``../../podaci/adjusted_surprisal.csv``.

Notes
-----
- Requires the ``transformers``, ``torch`` and ``stanza`` packages.
  Run ``stanza.download('sr')`` once before first use.
- Module-level model / tokenizer / NLP pipeline are loaded eagerly.
"""

import math
import os
from difflib import SequenceMatcher

import pandas as pd
import stanza
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from information_metrics.parameter_estimations.text_similarity_utils import levenshtein_distance, orthographic_similarity, sequence_matcher

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', '..', 'podaci', 'target_sentences.csv')
VOCABULARY_PATH = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_cleaned.csv')
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'adjusted_surprisal.csv')

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
    sequence-mean of the model's hidden output for each, and
    returns the rescaled cosine ``0.5 * (cos + 1)``.

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


def extract_words_and_embeddings(subwords, subword_embeddings):
    """Aggregate GPT-2 BPE sub-token embeddings into word embeddings.

    See :mod:`information_value` for a longer description.

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


def calculate_word_information_values(sentence, vocabulary_df, similarity_function, model = model, tokenizer = tokenizer):
    """Per-word adjusted surprisal under four similarity functions.

    For each word slot in ``sentence``, draws a 50-row
    ``random_state=42`` sample of ``vocabulary_df``, substitutes
    each candidate into the slot, and accumulates four
    context-probability-weighted similarities against the original
    word (contextual embedding, non-contextual embedding, POS-tag
    match, orthographic similarity). After the candidate sweep,
    each accumulator is converted to ``-log2(acc + 1e-40)`` to give
    a binary-log adjusted surprisal.

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
    ce_iv, nce_iv, pt_iv, od_iv : list of float
        Adjusted surprisal per slot under contextual embedding,
        non-contextual embedding, POS-tag and orthographic
        similarity respectively.
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

        ce_iv[i] += 0.5 * (F.cosine_similarity(embedding_word1, embedding_word2, dim=0).item() + 1) * context_probability
        nce_iv[i] += non_context_embedding(word, vord, model, tokenizer) * context_probability
        pt_iv[i] += pos_tags_similarity(word, vord, sentence, i) * context_probability
        od_iv[i] += orthographic_similarity(word, vord) * context_probability

      ce_iv[i] = - math.log(ce_iv[i] + 1e-40, 2)
      nce_iv[i] = - math.log(nce_iv[i] + 1e-40, 2)
      pt_iv[i] = - math.log(pt_iv[i] + 1e-40, 2)
      od_iv[i] = - math.log(od_iv[i] + 1e-40, 2)

    return words_list, ce_iv, nce_iv, pt_iv, od_iv


def main():
    """Build and save the per-word adjusted-surprisal table.

    Reads :data:`TARGET_SENTENCES_PATH` and :data:`VOCABULARY_PATH`,
    iterates target sentences from index ``23`` onwards (preserved
    verbatim from the source notebook), computes per-word adjusted
    surprisals via :func:`calculate_word_information_values` using
    :func:`pos_tags_similarity` as the similarity hook, and writes
    the incremental ``(Sentence, Word, AS Context Embedding,
    AS Non-context Embedding, AS Pos-Tag, AS Orthographic)`` table
    to :data:`OUTPUT_CSV_PATH` after each sentence.
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

    for i in range(23,len(target_sentences_df)):
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
          'AS Context Embedding': ce_iv_list,
          'AS Non-context Embedding': nce_iv_list,
          'AS Pos-Tag': pt_iv_list,
          'AS Orthographic': od_iv_list
                         })
      df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
