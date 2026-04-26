# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:34:11 2024

@author: Jelena

import os
from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")



filenames = [
    '.gitattributes', 'config.json', 'generation_config.json', 'model-00001-of-00003.safetensors',
    'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors', 'model.safetensors.index.json',
    'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json'
        ]

model_id = 'gordicaleksa/YugoGPT'

for filename in filenames:
        downloaded_model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=HUGGING_FACE_API_KEY
        )
        print(downloaded_model_path)

Pipeline role
-------------
Computes a contextual-entropy information measure for every word in
every target sentence using the Serbian-language causal LM
``gordicaleksa/YugoGPT``. For each sentence the script substitutes a
random sample of vocabulary candidates (50 rows from
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``, fixed
``random_state=42``) into each word slot, queries the language model
to obtain the per-subword softmax probability of the candidate in
context, aggregates subword probabilities back to whole-word level
via :func:`extract_words_and_probabilities`, and accumulates the
binary-log entropy contribution per slot. The resulting table
(``Sentence``, ``Word``, ``Contextual Entropy Yugo``) is written to
``../podaci/information measurements parameters/yugo_entropy_data.csv``
and is one of several alternative information measures consumed by
the surprisal / regression analyses in
``Different information measurement parameters/``.
"""

import pandas as pd
import os
import torch
import math 
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'gordicaleksa/YugoGPT'
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_id)


def extract_words_and_probabilities(subwords, subword_probabilities):
    """Aggregate sub-word probabilities into whole-word probabilities.

    Walks ``subwords`` left to right and treats every token that
    starts with the SentencePiece word-boundary marker ``▁`` as the
    beginning of a new word. Sub-word probabilities are multiplied
    together within the same word, so the returned per-word
    probability equals ``prod(p_i)`` over the sub-words ``i`` that
    compose it. The leading ``▁`` is stripped from the surface form
    of each emitted word.

    Parameters
    ----------
    subwords : list of str
        Decoded SentencePiece sub-word strings, in left-to-right
        order. Sub-words that start a new word begin with the
        boundary marker ``▁``.
    subword_probabilities : sequence of float
        Probabilities aligned 1:1 with ``subwords``.

    Returns
    -------
    tuple
        ``(words, word_probabilities)`` -- two lists of equal length
        where ``word_probabilities[k]`` is the product of all
        sub-word probabilities that were merged to form
        ``words[k]``.
    """
    words = []
    word_probabilities = []

    current_word = ""
    current_probability = 1.0  # Initialize to 1.0 as we will multiply probabilities

    # Iterate through the subword probabilities
    for subword, probability in zip(subwords, subword_probabilities):
        # Check if the subword starts with the special token 'Ġ'
        if subword.startswith('▁'):
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

def calculate_contextual_entropy(sentence, tokenizer, model, vocabulary_df):
    """Estimate per-word contextual entropy with a causal LM.

    For each position in ``sentence``, draws a fixed sample of 50
    vocabulary candidates (``vocabulary_df.sample(n=50,
    random_state=42)``), substitutes each candidate into the slot,
    encodes the modified sentence with ``tokenizer``, queries
    ``model`` to obtain per-token logits, averages the softmax over
    sequence positions to get a single probability per sub-word,
    aggregates sub-word probabilities to the whole-word level via
    :func:`extract_words_and_probabilities`, and accumulates the
    binary-log entropy contribution
    ``-p * log2(p + 1e-40)`` over candidates.

    Parameters
    ----------
    sentence : str
        Whitespace-tokenised sentence whose words will be replaced
        slot by slot.
    tokenizer : transformers.PreTrainedTokenizer
        SentencePiece-compatible tokenizer matching ``model``.
    model : transformers.PreTrainedModel
        Causal language model returning ``logits``.
    vocabulary_df : pandas.DataFrame
        Reference vocabulary with a ``word`` column. Only a 50-row
        ``random_state=42`` sample is used per word slot.

    Returns
    -------
    tuple
        ``(words_list, entropy_list)`` -- the original words from
        ``sentence`` (in order) and their accumulated contextual
        entropy contribution from the candidate sweep.
    """
    # calulate information value for one sentence
    words_list = []
    entropy_list = []

    # loop through all words in sentence
    for i in range(0, len(sentence.split(' '))):

      words = sentence.split(' ')
      words_list.append(words[i])

      entropy_list.append(0)
      vocab_df = vocabulary_df.sample(n=50, random_state=42).reset_index(drop=True)

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


target_sentence_path = os.path.join('..', '..','podaci', 'target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentence_path)

vocabulary_path = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_cleaned.csv')
vocabulary_df = pd.read_csv(vocabulary_path)

words_list = []
target_sentence_list = []
entropy_list = []

for i in range(66,len(target_sentences_df)):
  sentence = target_sentences_df['Text'][i].lower()
  print(i)
  words, entropies = calculate_contextual_entropy(sentence.strip(), tokenizer, model, vocabulary_df)

  for word, entropy in zip(words, entropies):
    words_list.append(word)
    target_sentence_list.append(i)
    entropy_list.append(entropy)


# Create a DataFrame
df = pd.DataFrame({'Sentence': target_sentence_list, 'Word': words_list, 'Contextuual Entropy Yugo': entropy_list})

# Find the maximum value in the 'Surprisal Yugo' column
max_value = df['Surprisal Yugo'].max()
# Replace 0 values with the maximum value
df['Surprisal Yugo'] = df['Surprisal Yugo'].replace(0, max_value)

# Save the DataFrame to a CSV file
df = df.dropna()
csv_file_path = os.path.join('..','podaci','information measurements parameters', "yugo_entropy_data.csv") 
df.to_csv(csv_file_path, index=False)