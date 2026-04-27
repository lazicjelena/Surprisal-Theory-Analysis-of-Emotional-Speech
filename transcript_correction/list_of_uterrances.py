# -*- coding: utf-8 -*-
"""list_of_uterrances.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Bootstraps the corrected-transcript workflow in
``Transcript - correct/``. Recursively walks the raw
``../podaci/transcript`` tree, reads the textual transcript
prefix of every ``*.txt`` file (everything before the first
``
Word:`` segmentation marker, with the leading
``Transcript:`` prefix stripped), tokenizes it into sentences
with NLTK ``sent_tokenize`` and keeps the first sentence per
file. The collected ``(File, First Sentence)`` table is written
to ``../podaci/first_sentences.csv``. A manually curated subset
of these sentences seeds ``../podaci/target_sentences.csv``
(the canonical inventory used by ``transcription_alignment.py``
and downstream scripts) and the unmatched residue seeds
``../podaci/wrong_transcription.csv``.

"""
import os
import pandas as pd
from nltk import sent_tokenize 

#nltk.download('punkt')

def read_first_sentence(file_path):
    """Return the first transcript sentence from a per-utterance file.

    The file is expected to start with a textual transcript that
    is followed by a ``\nWord:``-prefixed segmentation block.
    Only the prefix (with any leading ``Transcript:`` token
    stripped) is tokenized via NLTK ``sent_tokenize`` and the
    first sentence is returned.

    Parameters
    ----------
    file_path : str
        Path to a transcript ``*.txt`` file.

    Returns
    -------
    str or None
        The first sentence of the textual transcript, or
        ``None`` when the prefix tokenizes to an empty list.
    """
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