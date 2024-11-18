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