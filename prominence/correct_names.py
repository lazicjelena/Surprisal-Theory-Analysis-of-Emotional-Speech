# -*- coding: utf-8 -*-
"""correct_names.py
Jelenina skripta
lazic.jelenaa@gmail.com

Pipeline role
-------------
Cleanup pass over the renamed prosody folder
``../podaci/prosody``. Locates every ``*.prom`` file whose name
does not already follow the canonical
``<speaker>_<emotion>_<name>.prom`` convention (i.e. the first
four characters are not digits followed by an underscore) and
restores the canonical form by cross-referencing the original
``..\podaci\data_mono\<speaker>\<emotion>\<name>.wav``
hierarchy. Run once after
``move_data_to_final_folder.py`` has flattened the wavelet GUI
output.

"""

import os
import pandas as pd

def get_prom_files(folder):
    """Return ``.prom`` files in ``folder`` not following the canonical name.

    A canonical prosody file name starts with four digits
    (the speaker id) followed by an underscore. Files that do
    not satisfy this prefix pattern are returned for renaming.

    Parameters
    ----------
    folder : str
        Path to the prosody output directory.

    Returns
    -------
    list of str
        File names (basenames, not full paths) that need
        renaming.
    """
    prom_files = []
    for file in os.listdir(folder):
        if file.endswith(".prom") and not (file[:4].isdigit() and file[4] == '_'):
            prom_files.append(file)
    return prom_files

prom_dir = os.path.join('..','podaci','prosody')
prom_files = get_prom_files(prom_dir)

def collect_wav_files_info(data_folder):
    """Recursively collect every ``.wav`` path under ``data_folder``.

    Used to recover the canonical ``(speaker, emotion, name)``
    triple from the original audio directory tree, since the
    prosody output folder loses that information when the wavelet
    GUI flattens it.

    Parameters
    ----------
    data_folder : str
        Top-level audio directory (typically
        ``../podaci/data_mono``).

    Returns
    -------
    list of str
        Full paths of every WAV file found, in :func:`os.walk`
        order.
    """
    wav_files_info = []

    # Iterate through all subfolders and files in data_folder
    for root, dirs, files in os.walk(data_folder):
        # Iterate through all files in the current directory
        for file in files:
            # Check if the file is a .wav file
            if file.endswith('.wav'):
                # Get the full path of the .wav file
                file_path = os.path.join(root, file)
                # Append file information (path and subfolders) to the list
                wav_files_info.append((file_path))

    return wav_files_info

data_folder = os.path.join('..','podaci','data_mono')
wav_files_info = collect_wav_files_info(data_folder)

speaker = []
emotion = []
name = []

# Extracting speaker, emotion, and name from each file path
for path in wav_files_info :
    parts = path.split('\\')
    speaker.append(parts[-3])
    emotion.append(parts[-2])
    name.append(parts[-1].split('.')[0])  # Remove the '.wav' extension

# Creating a dataframe
df = pd.DataFrame({'speaker': speaker, 'emotion': emotion, 'name': name})


# Rename files
for file in prom_files:
    row = df[df['name']==file[:-5]]
    old_name = os.path.join(prom_dir, file)
    new_name = os.path.join(prom_dir, f"{row['speaker'].values[0]}_{row['emotion'].values[0]}_{row['name'].values[0]}.prom")
    os.rename(old_name, new_name)


    
    