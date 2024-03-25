# -*- coding: utf-8 -*-
"""list_of_uterrances.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skripta vrsi izdvajanje prve recenice iz svakog trnasripta. Na osnovu 
izdvojenih recenica napravljen spisak target recenica i wrong transription spisak.

"""
import os
import pandas as pd
from nltk import sent_tokenize 

#nltk.download('punkt')

def read_first_sentence(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # Split the content based on '\nWord:'
        parts = content.split('\nWord:')
        if len(parts) > 0:
            transcript_text = parts[0].strip()
        else:
            transcript_text = content.strip()

        # Remove "Transcript:" from the beginning of each sentence
        transcript_text = transcript_text.replace('Transcript:', '')

        sentences = sent_tokenize(transcript_text)
        if sentences:
            return sentences[0]
        else:
            return None

# Example usage
root_folder = os.path.join('..','podaci','transcript')

data = {'File': [], 'First Sentence': []}

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.lower().endswith(".txt"):
            file_path = os.path.join(root, file)
            first_sentence = read_first_sentence(file_path)

                # If the file contains at least one sentence
            if first_sentence is not None:
                data['File'].append(file_path)
                data['First Sentence'].append(first_sentence)

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_path = os.path.join('..','podaci','first_sentences.csv')
df.to_csv(output_path, index=False)