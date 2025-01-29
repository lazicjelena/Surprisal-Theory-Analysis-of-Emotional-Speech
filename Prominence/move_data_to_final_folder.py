# -*- coding: utf-8 -*-
"""move_data_to_final_folder.py

Jelenina skripta
lazic.jelenaa@gmail.com

Kada se racuna prosody parametar koristi se wavelet_gui.py koji automatski obradjuje
sve fajlove koji se nalaze u jednom folderu, zbog toga je prije upotrebe ovog programa
bilo potrebno prebaciti sve fajlove na istu lokaciju, a podataka o tome odakle su prebaceni,
odnosno ko je govornik i koja emocija je u pitanju, cuvan je u izmjenjenom nazivu fajla.
"""

import os
import shutil

current_folder_path = os.path.join('..', '..', 'wavelet_prosody_toolkit', 'data_prosody','all_files')
final_folder_path = os.path.join('..','podaci','prosody')

# Ensure the final folder exists, if not, create it
os.makedirs(final_folder_path, exist_ok=True)

# Get a list of all files in the current folder
file_list = os.listdir(current_folder_path)

# Filter out only the .prom files
prom_files = [file for file in file_list if file.endswith('.prom')]

# Move each .prom file to the final folder
for prom_file in prom_files:
    source_path = os.path.join(current_folder_path, prom_file)
    destination_path = os.path.join(final_folder_path, prom_file)
    shutil.move(source_path, destination_path)

print("Prom files moved successfully!")

# Loop through each .prom file
for prom_file in prom_files:
    # Get the base file name (without extension) of the prom file
    base_name = os.path.splitext(prom_file)[0]

    # Delete corresponding .wav and .lab files
    for ext in ['.wav', '.lab']:
        file_to_delete = os.path.join(current_folder_path, base_name + ext)
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            print(f"Deleted {file_to_delete}")

print("Corresponding .wav and .lab files deleted from the current folder!")