# -*- coding: utf-8 -*-
"""text_utils.py
Lexical/tekst pomocne funkcije izdvojene iz:
  - Prominence/librosa_estimated_parameters.py
  - Prominence/prominence_build_dataset.py

P-008 (Faza 2-B): zajednicke IDENTICNO funkcije unutar foldera
'Prominence/'. Tijelo funkcija NIJE mijenjano.

Napomena: find_subword postoji i u:
  - Additional files after recension/build_prominence_datasets.py
  - Pervious Surprisals/prominence_build_dataset.py
Cross-folder konsolidacija nije dio P-008 - ostaje za P-009.
"""


def find_subword(word, unique_words):

    subword = ''
    for i in range(1,len(word)+1):
        if word[-i:] in unique_words:
            subword = word[-i:]

    return subword
