# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:45:12 2024

@author: Jelena
"""
import torch.nn.functional as F
import torch
import math

''' Funkcije'''

def non_context_embedding(word, vord, model, tokenizer, j):

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