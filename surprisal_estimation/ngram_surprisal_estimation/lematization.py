# -*- coding: utf-8 -*-
"""lematization.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Lemmatisation step of the n-gram-language-model training pipeline.
Reads a Serbian-language text corpus
``../podaci/ngram datasets/nq_open.csv`` plus an incremental
checkpoint ``lemmas.csv`` (so the script can be resumed after an
interruption), runs every sentence through a CLASSLA Serbian
``tokenize, lemma, pos`` pipeline, and writes the per-token lemma
sequence back into the original CSV under a new ``lemma`` column.
A periodic checkpoint to ``lemmas.csv`` is written every 1000
sentences. The lemmatised corpus produced here, combined with
output from other ``ngram datasets/*.csv`` files, is the training
data for ``surprisal_estimation_n_gram_model.py``.

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