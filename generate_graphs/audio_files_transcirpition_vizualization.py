# -*- coding: utf-8 -*-
"""audio_files_transcirpition_vizualization.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skript vrsi vremenski prikaz jednog audio signala i njegove segmentacije i transkripcije
na nivou rijeci.

Pipeline role
-------------
Auxiliary visualisation script used to qualitatively inspect the
forced-alignment output. For a chosen speaker / emotion folder under
``../podaci/data_mono/<speaker>/<emotion>/`` it loads each WAV file,
plots the raw waveform, and overlays the per-word ``start`` / ``end``
markers (plus word labels) parsed from the matching
``..._transcript.txt`` produced by
``Forced alignment/novosadska_baza_podataka.py``. Not part of the
automated processing chain; strictly a debugging / figure-generation
helper.
"""

import wave
import numpy as np
import matplotlib.pyplot as plt
import os

# Definirajte rječnik koji mapira latinična slova na ćirilična
latinica_to_cirilica = {
    'a': 'а', 'b': 'б', 'c': 'ц', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
    'h': 'х', 'i': 'и', 'j': 'ј', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
    'o': 'о', 'p': 'п', 'q': 'љ', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
    'v': 'в', 'w': 'ш', 'x': 'ч', 'y': 'џ', 'z': 'з', 'č':'ч' , 'ć':'ћ' ,
    'š':'ш', 'ž':'ж'  
}

# Funkcija za konverziju stringa
def latinica_u_cirilicu(latinicni_string):
    """Transliterate a Serbian Latin string into Serbian Cyrillic.

    Walks ``latinicni_string`` character by character and replaces each
    letter with its Cyrillic counterpart from the module-level
    ``latinica_to_cirilica`` mapping. Characters that are not in the
    mapping (digits, punctuation, whitespace, etc.) are passed through
    unchanged. Lookup is case-insensitive: the lowercase form of the
    character is used as the dictionary key, but the resulting Cyrillic
    letter is taken from the mapping as-is.

    Parameters
    ----------
    latinicni_string : str
        Input string written in Serbian Latin script (with diacritics
        ``č``, ``ć``, ``š``, ``ž`` supported).

    Returns
    -------
    str
        The same string transliterated to Serbian Cyrillic.
    """
    cirilicni_string = ''
    for slovo in latinicni_string:
        # Ako je slovo u rječniku, zamijenite ga ćiriličnim ekvivalentom
        if slovo.lower() in latinica_to_cirilica:
            cirilicni_string += latinica_to_cirilica[slovo.lower()]
        else:
            cirilicni_string += slovo
    return cirilicni_string

def plot_single_transcript(audio_file_path, transcript_file_path):
    """Plot a WAV waveform with overlaid word-level transcript markers.

    Opens ``audio_file_path`` with :mod:`wave`, decodes the samples as
    8-bit unsigned or 16-bit signed PCM, draws the time-domain
    waveform, and overlays one dashed red line at every word ``start``
    time and one dashed green line at every word ``end`` time, with
    the word label placed at the midpoint above the signal. Word
    annotations are read from the matching transcript file produced
    by ``Forced alignment/novosadska_baza_podataka.py``.

    Parameters
    ----------
    audio_file_path : str
        Path to a single WAV recording.
    transcript_file_path : str
        Path to the matching ``..._transcript.txt`` file. Expected
        format: line 1 is ``"Transcript: <full sentence>"`` (skipped
        when overlaying markers); subsequent lines look like
        ``"Word: <w>, start: <float>, end: <float>"``.

    Returns
    -------
    None

    Side effects
    ------------
    Calls ``matplotlib.pyplot.show`` to display the figure. Raises
    ``ValueError`` if the WAV file uses a sample width other than 1
    or 2 bytes per sample.
    """

    # Open the WAV file
    with wave.open(audio_file_path, 'rb') as wav_file:
        # Get basic information about the WAV file
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
    
        # Read audio data (all frames)
        audio_data = wav_file.readframes(num_frames)
    
    # Convert binary audio data to a numpy array
    sample_width = wav_file.getsampwidth()
    if sample_width == 1:
        audio_array = np.frombuffer(audio_data, dtype=np.uint8)
    elif sample_width == 2:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    else:
        raise ValueError("Unsupported sample width")
    
    # Calculate time array
    time_array = np.arange(0, num_frames) / frame_rate
    # Plot audio data
    plt.figure(figsize=(10, 4))
    plt.plot(time_array, audio_array, color='b')
    plt.ylabel('амплитуда', fontsize = 15)
    plt.xlabel('вријеме (s)', fontsize = 15)
    plt.title('Аудио сигнал и његова транскрипција', fontsize = 15)
    
    # Read the transcript file
    with open(transcript_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Extract words, start and end times from the transcript
    for line in lines[1:]:  # Skipping the first line which contains "Transcript:"
        parts = line.strip().split(', ')
        word = parts[0].split(': ')[1]
        start_time = float(parts[1].split(': ')[1])
        end_time = float(parts[2].split(': ')[1])
        
        # Add vertical lines for start and end times
        plt.axvline(x=start_time, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=end_time, color='g', linestyle='--', alpha=0.5)
        
        # Annotate the graph with the words
        plt.text((start_time + end_time) / 2, max(audio_array), word, ha='center', va='center', fontsize = 15)
        
    plt.show()
    
    return


audio_folder_path = os.path.join('..','podaci', 'data_mono', '0001', '0') 
transcript_folder_path = os.path.join('..','podaci', 'transcript_corrected', '0001', '0') 

# Iterate over all files in the folder
for file_name in os.listdir(audio_folder_path):
    # Check if the file is a .wav file
    if file_name.endswith('.wav'):
        audio_file_path = os.path.join(audio_folder_path, file_name)
        transcript_file_path = os.path.join(transcript_folder_path, file_name[:-4] + '_transcript.txt')
        # Print the name of the .wav file
        try:
            plot_single_transcript(audio_file_path, transcript_file_path)
        except:
            continue