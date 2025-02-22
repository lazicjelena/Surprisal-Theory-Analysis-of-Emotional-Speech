# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:54:56 2025

@author: Jelena
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