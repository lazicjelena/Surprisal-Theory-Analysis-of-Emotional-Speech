# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:13:42 2024

@author: Jelena

Pretvara transkript fajlove u format koji odgovara wavele_gui aplikaciji.
"""
import os

def read_transcript_file(file_path):
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
    for word_info in transcript:
        word_info['start'] = int(word_info['start'] * 10000)
        word_info['end'] = int(word_info['end'] * 10000)
    return transcript

def write_lab_file(transcript, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:  # Specify encoding as UTF-8
        for word_info in transcript:
            start_ms = int(word_info['start'] * 1000)  # Convert start time to milliseconds
            end_ms = int(word_info['end'] * 1000)  # Convert end time to milliseconds
            line = "{} {} {}\n".format(start_ms, end_ms, word_info['Word'])
            f.write(line)

def process_directory(directory):
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
        





