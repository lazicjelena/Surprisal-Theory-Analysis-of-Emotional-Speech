# -*- coding: utf-8 -*-
"""fonetic_features.py

Per-word phonetic feature counts for Serbian target sentences.
Converted from ``Fonetic features.ipynb`` (P-013-NB).

Pipeline role
-------------
Phonetic feature extractor for the
``information_metrics/parameter_estimations/`` chain. For every word
in every sentence in ``../../podaci/target_sentences.csv``, segments
the word into Serbian phonemes (handling the digraphs ``dž``, ``lj``,
``nj`` and a small set of numeric spell-outs), and counts the number
of phonemes that fall in each phonological class -- vowels, sonorants
(alveolar / palatal / bilabial / labio-dental), consonants (labial /
dental / palatal / velar) and the voiced / voiceless split. Two
derived columns ``konsonanti`` and ``zvucni`` are built as sums of
sub-classes. The result is written to
``../../podaci/fonetic_features1.csv``.

Notes
-----
- The phonological class lists (``vokali``, ``sonanti``, etc.) are
  declared at module level using the original Serbian / Croatian
  variable names from the notebook so that the
  ``[name for name in globals() if globals()[name] is foneme_type]``
  lookup inside :func:`main` continues to work.
- The notebook re-defines the variable name ``palatalni`` (sonorant
  palatals first, then consonant palatals); the second binding
  shadows the first. Preserved verbatim from the source notebook.
"""

import os
import re

import pandas as pd

# Paths (relative to project root).
TARGET_SENTENCES_PATH = os.path.join('..', '..', 'podaci', 'target_sentences.csv')
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'fonetic_features1.csv')

# podjela po mjestu tvorbe
vokali = ['a', 'e', 'i', 'o', 'u']
sonanti = ['v', 'r', 'l', 'n', 'j', 'lj', 'nj', 'm']

# sonanti
alveolarni = ['r', 'l', 'n']
palatalni = ['j', 'lj', 'nj']
bilabijalni = ['m']
labio_dentalni = ['v']

# konsonanti
labialni = ['b', 'p', 'f']  # usneni
zubni = ['d', 't', 'z', 's', 'c']
palatalni = ['dž', 'č', 'đ', 'ć', 'ž', 'š']  # prednjenepcani
zadnjonepcani = ['k', 'g', 'h']  # velarni

# podjela po zvucnosti
zvucni = ['b', 'g', 'd', 'đ', 'ž', 'z', 'dž']
bezvucni = ['p', 'k', 't', 'ć', 'š', 's', 'č', 'h', 'f', 'c']

fonem_types = [vokali, sonanti, alveolarni, palatalni, bilabijalni, labio_dentalni, labialni, zubni, palatalni, zadnjonepcani, zvucni, bezvucni]


def extraxt_foneme(word):
    """Segment ``word`` into Serbian phonemes (digraph-aware).

    First special-cases a small set of numeric strings (``"10"``,
    ``"27"``, ``"50"``, ``"100"``) and replaces them with their
    Serbian spell-outs. Then applies the regex ``dz|lj|nj|.`` so
    that the digraphs ``dz``, ``lj`` and ``nj`` are kept as one
    phoneme each, while every other character forms a single-phoneme
    token. Note: the source notebook uses ``dz`` (Latin) in the
    pattern and ``dž`` (with caron) in the consonant tables;
    preserved verbatim here.

    Parameters
    ----------
    word : str
        Surface form of a single word (lower-cased upstream).

    Returns
    -------
    list of str
        The phoneme tokens that compose ``word``, in left-to-right
        order.
    """

    pattern = r'^[+-]?\d+(\.\d+)?$'
    if bool(re.match(pattern, word)):
      if word == '10':
        word = 'deset'
      if word == '27':
        word = 'dvadesetsedam'
      if word == '50':
        word = 'pedeset'
      if word == '100':
        word = 'sto'

    # Define the pattern to match 'dz' as one unit or any single letter
    pattern = r'dz|lj|nj|.'
    # Use findall to get all matches
    parts = re.findall(pattern, word)
    return parts


def main():
    """Build and save the per-word phonetic feature table.

    Reads :data:`TARGET_SENTENCES_PATH`, expands every sentence
    into one row per whitespace-separated word, then for each
    phonological class in :data:`fonem_types` counts how many
    phonemes (as returned by :func:`extraxt_foneme`) fall in that
    class and stores the count under a column named after the
    class's variable name. Finally, ``konsonanti`` and ``zvucni``
    are recomputed as sums of sub-classes (preserved verbatim from
    the notebook -- the second assignment overwrites the
    voiced-only ``zvucni`` column with a vowel + sonorant + voiced
    sum). The result is written to :data:`OUTPUT_CSV_PATH`.
    """
    target_sentences_df = pd.read_csv(TARGET_SENTENCES_PATH)

    sentence_list = []
    words_list = []

    for i in range(0,len(target_sentences_df)):

        sentence = target_sentences_df['Text'][i].lower()
        words = sentence.split(' ')

        for word in words:
          sentence_list.append(i)
          words_list.append(word)

    data = pd.DataFrame({'sentence':sentence_list, 'word':words_list})

    for foneme_type in fonem_types:
      fonem_list = []

      for index, row in data.iterrows():
        word = row['word']
        counter = 0
        foneme = extraxt_foneme(word)

        for fonem in foneme:
          if fonem in foneme_type:
            counter += 1

        fonem_list.append(counter)

      list_name = [name for name in globals() if globals()[name] is foneme_type][0]
      data[list_name] = fonem_list

    # Sum specific columns and create a new column 'sum_col'
    data['konsonanti'] = data[['labialni', 'zubni', 'palatalni', 'zadnjonepcani']].sum(axis=1)
    data['zvucni'] = data[['zvucni', 'vokali', 'sonanti']].sum(axis=1)

    data.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
