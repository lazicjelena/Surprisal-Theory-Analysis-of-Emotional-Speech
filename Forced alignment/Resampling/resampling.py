# -*- coding: utf-8 -*-
"""resampling.py

Jelenin fajl
lazic.jelenaa@gmail.com

Scripta vrsi promjena ucestanosti odabiranja jednog audio signala.
Samo jedan snimak ima ucestanost odabiranja duplo vecu nego svi ostali snimci.
Da bi se moglo iyvrsiti transkriptovanje snimka potrebno je prvo promjeniti ucestanost
odabiranja i snimiti ga sa novom ucestanoscu.

"""

import os
from pydub import AudioSegment

def resample_wav(input_file, output_file, target_sampling_rate):
    audio = AudioSegment.from_wav(input_file)
    resampled_audio = audio.set_frame_rate(target_sampling_rate)
    resampled_audio.export(output_file, format="wav")

def resample_files_in_directory(root_folder, target_sampling_rate):
    # Create a new directory for the resampled files
    resampled_folder = os.path.join(root_folder[:-5], '1052_Resampled')
    os.makedirs(resampled_folder, exist_ok=True)

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                input_file_path = os.path.join(root, file)

                # Extract the folder name (0, 1, 2, 3, 4)
                folder_name = os.path.basename(root)

                # Create a folder in 1052_Resampled if it doesn't exist
                output_folder = os.path.join(resampled_folder, folder_name)
                os.makedirs(output_folder, exist_ok=True)

                # Construct the output file path
                output_file_path = os.path.join(output_folder, file)

                # Resample the WAV file
                resample_wav(input_file_path, output_file_path, target_sampling_rate)

# Example usage
root_folder = os.path.join('..','..','podaci', 'data_mono', '1052')
target_sampling_rate = 44100  # Desired sampling frequency

resample_files_in_directory(root_folder, target_sampling_rate)