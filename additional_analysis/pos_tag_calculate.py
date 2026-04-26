# -*- coding: utf-8 -*-
"""pos_tag_calculate.py

Created on Fri Feb 21 21:54:56 2025

@author: Jelena

Pipeline role
-------------
Per-word part-of-speech tag table builder. Initialises the
Serbian Stanza pipeline (``stanza.Pipeline('sr')``), walks the
canonical ``../podaci/target_sentences.csv`` inventory and,
via :func:`get_pos`, assigns a Universal POS tag to every
distinct word. The deduplicated ``(word, pos tag)`` table is
written to ``../podaci/pos_tags.csv`` and consumed by the
per-word analyses downstream.
"""

import pandas as pd
import stanza
import os

# Initialize the Stanza pipeline for Serbian
#stanza.download('sr')  # Ensure the model is downloaded
nlp = stanza.Pipeline('sr')

ts_file_path =  os.path.join('..','podaci', 'target_sentences.csv')
output_file_path =  os.path.join('..','podaci', 'pos_tags.csv')
df = pd.read_csv(ts_file_path)

def get_pos(word):
    """Return the Universal POS tag of ``word`` from the Serbian Stanza pipeline.

    Runs the module-level ``nlp`` Stanza pipeline on the input
    word and returns the ``upos`` attribute of the first
    word of the first sentence in the resulting document. Used
    by ``pos_tag_calculate.py`` to build a per-word
    grammatical-type inventory.

    Parameters
    ----------
    word : str
        Single word to tag.

    Returns
    -------
    str
        Universal POS tag (e.g. ``"NOUN"``, ``"VERB"``).
    """
    doc = nlp(word)  # Process the word with Stanza
    return doc.sentences[0].words[0].upos  # Return the Universal POS tag

# Example usage
pos_tag_list = []
word_list = []
pos_tag_list = []

for _, row in df.iterrows():
    
    words = row['Text'].lower().strip() # Example word: "dog"
    
    for word in words.split(' '):
        
        if word not in word_list:
            pos_tag = get_pos(word)
            pos_tag_list.append(pos_tag)
            word_list.append(word)
    
# Create DataFrame
df_grouped = pd.DataFrame({
    "word": word_list,
    "pos tag": pos_tag_list,
})

df_grouped.to_csv(output_file_path, index=False)