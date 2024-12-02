# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:49:49 2024

@author: Jelena
"""

#pip install transformers

from information_and_distance_functions import  calculate_word_information_values
import pandas as pd
import os

target_sentence_path = os.path.join('..','podaci','target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentence_path)

vocabulary_path = os.path.join('..','podaci','wordlist_classlawiki_sr_cleaned.csv')
vocabulary_df = pd.read_csv(vocabulary_path)

''' Load model '''
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'gpt2'
# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


''' Main '''

words_list = []
target_sentence_list = []
ce_iv_list = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
nce_iv_list = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

# Save the DataFrame to a CSV file
csv_file_path = os.path.join('..','podaci','information measurements parameters', "embedding_information_value1.csv")
                             

for i in range(46,50):
  sentence = target_sentences_df['Text'][i].lower()
  print(i)
  words, ce_ivs, nce_ivs = calculate_word_information_values(sentence.strip(), vocabulary_df, model, tokenizer)

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

# Display the DataFrame
print(df)
