# -*- coding: utf-8 -*-
"""lematization.py

Jelenina skripta
lazic.jelenaa@gmail.com

Ovdje se vrsi lematizacija skupa podataka nad kojim ce se vrsiti estimacija 
ngram modela.
Prolazi se kroz nekoliko skupova podataka, rezultat su .csv fajlovi koji sadrze
recenice i lematizaciju rijeci iz recenica. Na ove skupove podataka primjenjuje
ngram model.

"""


import pandas as pd
import classla
import os

#classla.download('sr')

nlp = classla.Pipeline('sr', processors='tokenize, lemma, pos')

data_file_path = os.path.join('..','podaci', 'ngram datasets', 'nq_open.csv')
data_df = pd.read_csv(data_file_path)
lemmas_file_path = os.path.join('..','podaci', 'ngram datasets', 'lemmas.csv')
lemmas_df = pd.read_csv(lemmas_file_path)

words_lemma_list = [lemmas for lemmas in lemmas_df['0']]
print(len(data_df))
ind = 0

for sentence in data_df['text'][len(words_lemma_list):]:
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
  ind += 1
  if not ind%100: 
      print(ind)
  if not ind%1000:
      lemmas_df = pd.DataFrame(words_lemma_list)
      lemmas_df.to_csv('lemmas.csv', index=False)


data_df['lemma'] = words_lemma_list

data_df.to_csv(data_file_path, index=False)