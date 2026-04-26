# -*- coding: utf-8 -*-
"""convert_txt_to_lib.py

Created on Thu Apr 25 18:13:42 2024

@author: Jelena

Pretvara transkript fajlove u format koji odgovara wavele_gui aplikaciji.

Pipeline role
-------------
First step of the wavelet-prosody preprocessing chain in
``Prominence/``. Walks
``../../wavelet_prosody_toolkit/data_prosody`` recursively, parses
every ``*_transcript.txt`` produced by the forced-alignment stage
and writes a sibling ``.lab`` file in the time-stamped 100-ns
units expected by ``wavelet_gui.py``. Output ``.lab`` files are
then renamed and merged into a single ``all_files`` directory by
``organize_data_for_wavelet_gui.py``.
"""
import os

def read_transcript_file(file_path):
    """Parse a forced-alignment transcript into per-word entries.

    Reads ``file_path`` line by line, keeping only lines beginning
    with ``"Word:"``. The expected line format is
    ``Word: <w>, start: <s>, end: <e>`` where the spoken word may
    contain spaces (which are stripped from the parsed token).
    Times are taken as plain floats (seconds).

    Parameters
    ----------
    file_path : str
        Path to a ``*_transcript.txt`` file.

    Returns
    -------
    list of dict
        One dict per word with keys ``Word``, ``start`` and
        ``end``.
    """
    transcript = []
    with open(file_path, "r", encoding="utf-8") as file:  # Specify encoding as UTF-8
        lines = file.readlines()
        for line in lines:
            if line.startswith("Word:"):
                parts = line.strip().split(", ")
                word = parts[0].split(": ")[1]
                word = word.replace(' ', '')
                start = float(parts[1].split(": ")[1])
                end = float(parts[2].split(": ")[1])
                transcript.append({"Word": word, "start": start, "end": end})
    return transcript

def time_convert(transcript):
    """Convert ``start``/``end`` from seconds to 100 microsecond units.

    Each ``start`` and ``end`` is multiplied by ``10000`` and cast
    to ``int``, in place.

    Parameters
    ----------
    transcript : list of dict
        Output of :func:`read_transcript_file`.

    Returns
    -------
    list of dict
        The same list with mutated ``start``/``end`` values.
    """
    for word_info in transcript:
        word_info['start'] = int(word_info['start'] * 10000)
        word_info['end'] = int(word_info['end'] * 10000)
    return transcript

def write_lab_file(transcript, output_file):
    """Write a ``.lab`` file in the wavelet GUI's expected format.

    Each line of the output file has the form
    ``"<start_ms> <end_ms> <Word>"`` where ``start_ms`` and
    ``end_ms`` are obtained by multiplying the input
    ``start``/``end`` (already in 100 microsecond units after
    :func:`time_convert`) by ``1000``.

    Parameters
    ----------
    transcript : list of dict
        Output of :func:`time_convert`.
    output_file : str
        Path of the ``.lab`` file to write.

    Returns
    -------
    None
    """
    with open(output_file, 'w', encoding='utf-8') as f:  # Specify encoding as UTF-8
        for word_info in transcript:
            start_ms = int(word_info['start'] * 1000)  # Convert start time to milliseconds
            end_ms = int(word_info['end'] * 1000)  # Convert end time to milliseconds
            line = "{} {} {}\n".format(start_ms, end_ms, word_info['Word'])
            f.write(line)

def process_directory(directory):
    """Recursively convert every ``.txt`` transcript under ``directory`` to ``.lab``.

    For every file with a ``.txt`` extension, runs
    :func:`read_transcript_file`, :func:`time_convert` and
    :func:`write_lab_file`; the output ``.lab`` is placed next to
    the source file with the trailing ``"_transcript"`` suffix
    stripped from the basename. Directories are recursed into.

    Parameters
    ----------
    directory : str
        Top-level directory to process.

    Returns
    -------
    None
    """
    # Loop through each file in the folder
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            transcript = read_transcript_file(file_path)
            transcript = time_convert(transcript)

            output_file = os.path.splitext(file_path)[0][:-11] + '.lab'
            write_lab_file(transcript, output_file)

        elif os.path.isdir(file_path):
            process_directory(file_path)  # Recursively process subdirectories

# Folder path to the top-level directory containing your data
top_folder_path = os.path.join('..', '..', 'wavelet_prosody_toolkit', 'data_prosody')

# Process the directory and its subdirectories
process_directory(top_folder_path)
        





