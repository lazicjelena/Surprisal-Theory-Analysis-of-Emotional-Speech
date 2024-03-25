# -*- coding: utf-8 -*-
"""audio_files_transcirpition_vizualization.py

Jelenina skripta
lazic.jelenaa@gmail.com

Skript vrsi vremenski prikaz jednog audio signala i njegove segmentacije i transkripcije
na nivou rijeci.
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
    cirilicni_string = ''
    for slovo in latinicni_string:
        # Ako je slovo u rječniku, zamijenite ga ćiriličnim ekvivalentom
        if slovo.lower() in latinica_to_cirilica:
            cirilicni_string += latinica_to_cirilica[slovo.lower()]
        else:
            cirilicni_string += slovo
    return cirilicni_string

def plot_single_transcript(audio_file_path, transcript_file_path):
    
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