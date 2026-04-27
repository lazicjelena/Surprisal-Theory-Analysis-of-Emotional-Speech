# -*- coding: utf-8 -*-
"""text_features_extraction.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Bridge between the raw forced-alignment output and the downstream
analysis stages. For every corrected transcript ``.txt`` (one per
recording, organised as ``transcript_corrected/<speaker>/<emotion>/``),
the script emits one row per spoken word into a per-emotion CSV under
``../podaci/data/<speaker>/<emotion>/<speaker>_<emotion>.csv`` with the
columns: ``word``, ``speaker``, ``speaker gender``, ``emotion``,
``length`` (number of characters), ``time`` (word duration in seconds),
``position`` (``b`` / ``m`` / ``e``) and ``taget sentence`` (index in
``target_sentences.csv``, matched fuzzy via FuzzyWuzzy ``fuzz.ratio``).
Speaker gender is read once from ``../podaci/speaker_gender.txt``.
The CSVs produced here are the input to virtually every downstream
``build_dataset.py`` and analysis script.


"""

import os
import pandas as pd
import warnings
from fuzzywuzzy import fuzz  

target_sentences_path = os.path.join('..','podaci', 'target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentences_path)

file_path = os.path.join('..','podaci', 'speaker_gender.txt')

try:
    # Read the text file into a DataFrame with custom column names
    speaker_gender_df = pd.read_csv(file_path, delimiter='\t', header=None, names=['combined'])

    # Extract speaker and gender using a regular expression
    speaker_gender_df[['speaker', 'gender']] = speaker_gender_df['combined'].str.extract(r'(\d+)\s*([mf])')

    # Drop the original combined column
    speaker_gender_df = speaker_gender_df.drop('combined', axis=1)

    print(speaker_gender_df)
except FileNotFoundError:
    print(f"The file at {file_path} could not be found.")
except pd.errors.EmptyDataError:
    print(f"The file at {file_path} is empty.")
except pd.errors.ParserError as pe:
    print(f"Error parsing the file at {file_path}: {pe}")
except Exception as e:
    print(f"An error occurred: {e}")

def find_target_sentence(sentence, df = target_sentences_df):
    """Find the row in ``df`` whose ``Text`` is most similar to ``sentence``.

    Iterates over every row of ``df`` and computes
    ``fuzzywuzzy.fuzz.ratio(sentence, row['Text'])``. The row with the
    highest ratio wins.

    Parameters
    ----------
    sentence : str
        Transcribed sentence (typically the first line of a corrected
        transcript ``.txt``).
    df : pandas.DataFrame, optional
        Reference table of canonical target sentences with a ``Text``
        column. Defaults to the module-level ``target_sentences_df``
        loaded from ``../podaci/target_sentences.csv``.

    Returns
    -------
    int
        Row index of the best-matching sentence in ``df``. Returns
        ``-1`` only if every comparison yields ratio ``0`` (in practice
        not expected on the corpus but kept as a safety value).
    """
    max_similarity = 0
    target_index = -1  # Initialize with an invalid index

    for index, row in df.iterrows():
        current_similarity = fuzz.ratio(sentence, row['Text'])
        if current_similarity > max_similarity:
            max_similarity = current_similarity
            target_index = index

    return target_index

def get_word_position(word_count):
    """Map a 1-based word count to a categorical position label.

    Used to label every word in a sentence with where it appears in the
    sentence. The mapping is fixed: word #1 is the beginning, word #2
    is the middle, and every later word is treated as the end.

    Parameters
    ----------
    word_count : int
        1-based index of the current word within the sentence.

    Returns
    -------
    str
        ``'b'`` (beginning) for ``word_count == 1``, ``'m'`` (middle)
        for ``word_count == 2``, otherwise ``'e'`` (end).
    """
    # Determine the word position based on the word count
    if word_count == 1:
        return 'b'  # Beginning of the sentence
    elif word_count == 2:
        return 'm'  # Middle of the sentence
    else:
        return 'e'  # End of the sentence

def process_txt_file(file_path):
    """Parse one transcript ``.txt`` file into per-word annotation dicts.

    Reads ``file_path``, takes the first line as the full transcript
    (skipping the ``"Transcript: "`` prefix, hence ``[12:]``), looks up
    the closest canonical sentence with :func:`find_target_sentence`,
    then walks every line that starts with ``"Word"`` and parses out
    the word, its ``start`` and ``end`` times, computes its
    ``position`` via :func:`get_word_position`, and attaches the
    target-sentence index.

    Parameters
    ----------
    file_path : str
        Path to a transcript ``.txt`` produced by the forced-alignment
        stage. Expected line format::

            Transcript: <full sentence>
            Word: <w>, start: <float>, end: <float>
            Word: <w>, start: <float>, end: <float>
            ...

    Returns
    -------
    list[dict]
        One dict per recognised word with keys
        ``'word'``, ``'start'``, ``'end'``, ``'position'`` and
        ``'target sentence'``.
    """
    # Add your code to process each TXT file here
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        index = find_target_sentence(lines[0][12:].strip())

        word_info_list = []
        total_words = 0
        for line in lines:
            if line.startswith('Word'):
                word_info = {}
                parts = line.split(', ')
                word_info['word'] = parts[0].split(': ')[1]
                word_info['start'] = float(parts[1].split(': ')[1])
                word_info['end'] = float(parts[2].split(': ')[1])

                # Determine word position in the sentence
                total_words += 1
                position = get_word_position(total_words)
                word_info['position'] = position
                word_info['target sentence'] = index

                word_info_list.append(word_info)

        return word_info_list

def calculate_word_length(word):
  """Total number of characters across whitespace-split tokens.

  Splits ``word`` on spaces and sums the character lengths of the
  resulting tokens, so that a multi-token expression (e.g.
  ``"za to"``) is counted as ``len("za") + len("to") == 4`` rather
  than as ``5`` (the length including the space).

  Parameters
  ----------
  word : str
      A single word or a whitespace-joined multi-word token.

  Returns
  -------
  int
      Sum of character lengths across the whitespace-split parts.
  """

  words = word.split(' ')
  l = 0
  for i in words:
    l+= len(i)

  return l

def get_gender(speaker):
  """Look up the gender label for a speaker id.

  Parameters
  ----------
  speaker : str
      Speaker identifier (typically a 4-character numeric string such
      as ``'1052'``) matching a value in
      ``speaker_gender_df['speaker']``.

  Returns
  -------
  str
      The associated gender label, ``'m'`` or ``'f'``. Raises
      ``IndexError`` (via ``.values[0]``) if the speaker is missing
      from ``speaker_gender_df``.
  """
  return speaker_gender_df.loc[speaker_gender_df['speaker'] == speaker, 'gender'].values[0]

def process_directory(directory_path, output_path):
    """Build a per-(speaker, emotion) feature CSV from one transcript folder.

    Iterates over every ``.txt`` file inside ``directory_path``, calls
    :func:`process_txt_file` to extract per-word annotations, enriches
    each row with ``speaker``, ``speaker gender``, ``emotion``,
    word ``length`` and word ``time`` (``end - start``), then writes
    the assembled DataFrame as
    ``<output_path>/<speaker>/<emotion>/<speaker>_<emotion>.csv``.

    The ``speaker`` value is derived from the path of
    ``directory_path``: the second-to-last folder name truncated to its
    first four characters. The ``emotion`` value is the last folder
    name of ``directory_path``.

    Parameters
    ----------
    directory_path : str
        Folder containing transcript ``.txt`` files for one
        (speaker, emotion) pair, e.g.
        ``../podaci/transcript_corrected/1052/0``.
    output_path : str
        Root output folder under which ``<speaker>/<emotion>/``
        subfolders will be created if missing, e.g.
        ``../podaci/data``.

    Returns
    -------
    None

    Side effects
    ------------
    - Creates the ``<output_path>/<speaker>/<emotion>/`` directory
      if it does not yet exist.
    - Writes one CSV file ``<speaker>_<emotion>.csv`` containing one
      row per spoken word (columns include ``word``, ``speaker``,
      ``speaker gender``, ``emotion``, ``length``, ``time``,
      ``position`` and ``taget sentence``).
    - Calls ``warnings.filterwarnings("ignore")`` at the start and
      ``warnings.resetwarnings()`` at the end to silence pandas
      ``FutureWarning`` / ``DeprecationWarning`` messages that
      ``DataFrame.append`` would otherwise raise inside the loop.
    """

    # Create an empty DataFrame with desired column names
    columns = ['word', 'speaker', 'emotion', 'time', 'position', 'taget sentence']
    df = pd.DataFrame(columns=columns)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Extract information about the two last folders
        speaker = os.path.split(directory_path)[0].split(os.path.sep)[-2:][-1][:4]
        emotion = os.path.split(directory_path)[1]

        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            words = process_txt_file(file_path)

            # Add rows to the DataFrame for each word
            for word_info in words:
                word, start, end, position, target = word_info['word'], word_info['start'], word_info['end'], word_info['position'], word_info['target sentence']
                word_length = calculate_word_length(word)
                gender = get_gender(speaker)
                df = df.append({
                    'word': word,
                    'speaker': str(speaker),
                    'speaker gender': gender,
                    'emotion': emotion,
                    'length': word_length,
                    'time': float(end) - float(start),
                    'position': position,
                    'taget sentence': target
                }, ignore_index=True)

    # Create speaker and emotion folders if they don't exist
    speaker_folder = os.path.join(output_path, speaker)
    emotion_folder = os.path.join(speaker_folder, emotion)
    os.makedirs(emotion_folder, exist_ok=True)

    # Save the DataFrame as a CSV file in the emotion folder
    print('Output folder: ')
    print(emotion_folder)
    output_file_path = os.path.join(emotion_folder, speaker + '_' + emotion + '.csv')
    df.to_csv(output_file_path, index=False)

     # Reset warnings filter to default (optional)
    warnings.resetwarnings()

    return

directory_path = os.path.join('..','podaci', 'transcript_corrected')
output_directory = os.path.join('..','podaci', 'data')

# Loop through all folders in the main directory
for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
    print('Processing: ')
    print(folder_path)

    # Check if the current item is a directory
    if os.path.isdir(folder_path):
        # Loop through subdirectories within the current directory
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)

            # Check if the current item in the subdirectory is a directory
            if os.path.isdir(subfolder_path):
                process_directory(subfolder_path, output_directory)