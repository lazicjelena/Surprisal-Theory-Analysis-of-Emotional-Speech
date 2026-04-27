# -*- coding: utf-8 -*-
"""cleaning_vocabulary_data.py

Vocabulary cleaning and Cyrillic-to-Latin normalisation for the
CLASSLA Wikipedia Serbian wordlist. Converted from
``Cleaning vocabulary data.ipynb`` (P-013-NB).

Pipeline role
-------------
Pre-processing step for the
``information_metrics/parameter_estimations/`` chain. Reads the raw
CLASSLA Wikipedia wordlist
``../../podaci/wordlist_classlawiki_sr_20240122114347.csv``, transliterates
any Cyrillic words to Latin via :func:`convert_cyrillic_to_latin`,
drops rows whose ``word`` is pure punctuation or starts with ``#``,
deduplicates by ``word``, and writes the cleaned result to
``../../podaci/wordlist_classlawiki_sr_cleaned.csv``. The cleaned
wordlist is consumed by the contextual-entropy and information-value
notebooks in the same folder (e.g. :mod:`gpt2_contextual_entropy`,
:mod:`information_value`).

Notes
-----
- The original notebook also rendered a histogram of word
  frequencies via ``matplotlib`` / ``seaborn``; that visualisation
  cell is not required for the pipeline output and is omitted from
  this script.
"""

import os

import pandas as pd

# Paths (relative to project root).
INPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_20240122114347.csv')
OUTPUT_CSV_PATH = os.path.join('..', '..', 'podaci', 'wordlist_classlawiki_sr_cleaned.csv')

# Character mapping from Cyrillic to Latin
cyrillic_to_latin = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i',
    'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm',
    'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r',
    'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f',
    'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
    'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I',
    'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M',
    'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R',
    'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F',
    'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š'
}


# Function to convert Cyrillic to Latin only if the text contains Cyrillic characters
def convert_cyrillic_to_latin(text):
    """Transliterate ``text`` from Serbian Cyrillic to Latin script.

    Checks whether ``text`` contains any character in the Cyrillic
    Unicode range and, if so, maps each character through
    :data:`cyrillic_to_latin`. Texts containing only Latin (or
    other non-Cyrillic) characters are returned unchanged.

    Parameters
    ----------
    text : str
        Word (or any string) potentially in Cyrillic.

    Returns
    -------
    str
        The Latin-script transliteration of ``text`` if Cyrillic
        characters were detected, otherwise ``text`` unchanged.
    """
    # Check if the text contains any Cyrillic characters
    if any('а' <= char <= 'я' or 'А' <= char <= 'Я' for char in text):
        return ''.join(cyrillic_to_latin.get(char, char) for char in text)
    return text  # Return the text unchanged if no Cyrillic characters are found


def main():
    """Build and save the cleaned Serbian wordlist.

    Reads :data:`INPUT_CSV_PATH`, casts ``word`` to ``str``,
    transliterates Cyrillic entries to Latin via
    :func:`convert_cyrillic_to_latin`, drops rows whose ``word``
    starts with punctuation or ``#``, deduplicates by ``word``,
    and writes the result to :data:`OUTPUT_CSV_PATH`.
    """
    df = pd.read_csv(INPUT_CSV_PATH)

    # Convert 'word' column to string to ensure compatibility with regex
    df['word'] = df['word'].astype(str)

    # Apply the conversion function to the 'word' column
    df['word'] = df['word'].apply(convert_cyrillic_to_latin)

    # Filter out rows where 'word' contains only punctuation or starts with '#'
    df = df[~df['word'].str.match(r'^[.,#\W].*')]

    df = df.drop_duplicates(subset='word')

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_PATH, index=False)  # Set index=False to avoid saving the index column


if __name__ == "__main__":
    main()
