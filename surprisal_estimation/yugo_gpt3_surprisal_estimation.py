# -*- coding: utf-8 -*-
"""yugo_gpt3_surprisal_estimation.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Estimates per-word surprisal for every target sentence using the
Serbian-language causal LM ``gordicaleksa/YugoGPT``. For each row
of ``../podaci/target_sentences.csv`` it tokenizes the lower-cased
sentence, runs the model, averages the per-position softmax to get
sub-word probabilities, aggregates sub-words back to whole-word
probabilities via :func:`extract_words_and_probabilities`, and
emits ``-log2(p)`` as the per-word surprisal. The result table
(``Sentence``, ``Word``, ``Surprisal Yugo``) is written to
``../podaci/word_surprisals_yugo.csv``, which is the input to
``Pervious Surprisals/build_dataset.py`` and to all downstream
analysis scripts that read ``Surprisal Yugo``.

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

    Walks ``subwords`` left to right and starts a new word every
    time a token starts with the SentencePiece word-boundary
    marker ``▁``. Sub-word probabilities are multiplied within
    the same word, so the per-word probability equals
    ``prod(p_i)`` over the sub-words ``i`` that compose it. The
    leading ``▁`` is stripped from the surface form.

    Parameters
    ----------
    subwords : list of str
        SentencePiece sub-word strings, in left-to-right order.
    subword_probabilities : sequence of float
        Probabilities aligned 1:1 with ``subwords``.

    Returns
    -------
    tuple
        ``(words, word_probabilities)`` -- two parallel lists where
        ``word_probabilities[k]`` is the product of sub-word
        probabilities that compose ``words[k]``.
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

def calculate_word_probabilities(sentence, tokenizer = tokenizer, model = model):
    """Compute per-word probabilities for ``sentence`` under YugoGPT.

    Tokenizes ``sentence`` with ``tokenizer``, runs ``model`` once
    in ``torch.no_grad()`` mode to obtain ``logits``, takes the
    softmax over the vocabulary axis and averages over sequence
    positions (``mean(dim=1)``) to get a single probability per
    sub-word. Sub-word probabilities are then aggregated to
    whole-word probabilities via
    :func:`extract_words_and_probabilities`.

    Parameters
    ----------
    sentence : str
        Lower-cased sentence to score.
    tokenizer : transformers.PreTrainedTokenizer, optional
        Defaults to the module-level YugoGPT tokenizer.
    model : transformers.PreTrainedModel, optional
        Defaults to the module-level YugoGPT model.

    Returns
    -------
    tuple
        ``(words, probabilities, total_probability)`` where
        ``words`` and ``probabilities`` are produced by
        :func:`extract_words_and_probabilities` and
        ``total_probability`` is the running sum of sub-word
        probabilities (kept for diagnostic purposes; not used
        downstream).
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

    total_probability =  0
    subwords = []
    subwords_probabilities = []
    # Display word probabilities for each word
    for word, probability in zip(decoded_words, word_probabilities[0]):
      subwords.append(word)
      subwords_probabilities.append(probability)
      total_probability += probability

    words, probabilities = extract_words_and_probabilities(subwords, subwords_probabilities)

    return words, probabilities, total_probability


target_sentence_path = os.path.join('..','podaci', 'target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentence_path)

words_list = []
probabilities_list = []
target_sentence_list = []

for i in range(0,len(target_sentences_df)):
  sentence = target_sentences_df['Text'][i].lower()
  words = sentence.split(' ')
  _, probabilities, total = calculate_word_probabilities(sentence)

  for word, prob in zip(words, probabilities[1:]):
    words_list.append(word)
    try:
        probabilities_list.append(-math.log2(prob.item()))
    except:
        print(word)
        probabilities_list.append(math.log2(1))
    target_sentence_list.append(i)

# Create a DataFrame
df = pd.DataFrame({'Sentence': target_sentence_list, 'Word': words_list, 'Surprisal Yugo': probabilities_list})

# Find the maximum value in the 'Surprisal Yugo' column
max_value = df['Surprisal Yugo'].max()
# Replace 0 values with the maximum value
df['Surprisal Yugo'] = df['Surprisal Yugo'].replace(0, max_value)

# Save the DataFrame to a CSV file
df = df.dropna()
csv_file_path = os.path.join('..','podaci', 'word_surprisals_yugo.csv')
df.to_csv(csv_file_path, index=False)