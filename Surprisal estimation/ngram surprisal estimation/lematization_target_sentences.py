# -*- coding: utf-8 -*-
"""lematization.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ovdje se vrsi lematizacija target recenica za potrebe odredjivanja ngram modela
pojedinacnih rijeci. Jer je i skup podataka na kome je vrsena procjena ngrama
lematizovan.
"""

import pandas as pd
import classla
import os

classla.download('sr')

nlp = classla.Pipeline('sr', processors='tokenize, lemma, pos')

data_name = os.path.join('..', '..', 'podaci','target_sentences.csv')
data_df = pd.read_csv(data_name)

words_lemma_list = []

for sentence in data_df['Text']:
  sentence = sentence.lower()
  words = sentence.split(' ')

  lemma_list = []
  for word in words:
    if word == '':
      continue
    doc = nlp(word)     # run the pipeline
    lemma = doc.to_conll().split()[-8]
    lemma_list.append(lemma)
  words_lemma_list.append(' '.join(lemma_list))


data_df['lemma'] = words_lemma_list
data_name = os.path.join('..', '..', 'podaci','target_sentences_lemmas.csv')
data_df.to_csv(data_name, index=False)