# -*- coding: utf-8 -*-
"""organize_data_for_wavelet_gui.py

Created on Wed May  8 15:23:01 2024

@author: Jelena

Nakon sto se izvrsi pretvaranje .txt fajlova u .lib format pokrece se ovaj fajl
i finalni rezultat je organizacija svih .wav i .lib fajlova u all-files folderu.

Pipeline role
-------------
Second step of the wavelet-prosody preprocessing chain (after
``convert_txt_to_lib.py``). First strips the original
``*_transcript.txt`` files and renames every ``.wav``/``.lab``
file under ``../../wavelet_prosody_toolkit/data_prosody`` to the
canonical ``<speaker>_<emotion>_<name>.<ext>`` form (so that
provenance is preserved when the file is moved out of its
directory hierarchy), then flattens the entire tree into a single
``all_files`` directory so the wavelet GUI can batch-process the
whole dataset in one run.
"""

import os
import shutil

# Folder path to the top-level directory containing your data
top_folder_path = os.path.join('..', '..', 'wavelet_prosody_toolkit', 'data_prosody')

def rename_files_and_delete_txt_files(top_folder_path):
    """Rename ``.wav``/``.lab`` files to canonical form and delete ``.txt``.

    For every directory under ``top_folder_path``, deletes any
    ``.txt`` files (transcripts that have already been converted
    to ``.lab`` by ``convert_txt_to_lib.py``) and renames every
    ``.wav``/``.lab`` to
    ``<parent_folder>_<current_folder>_<old_name>.<ext>``, where
    ``<parent_folder>`` is the speaker id and
    ``<current_folder>`` is the emotion id.

    Parameters
    ----------
    top_folder_path : str
        Top-level wavelet prosody data directory.

    Returns
    -------
    None
    """
    for root, dirs, files in os.walk(top_folder_path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    file_name, file_ext = os.path.splitext(filename)
                    if file_ext == '.txt':
                        # Delete .txt files
                        os.remove(file_path)
                    elif file_ext in ['.lab', '.wav']:
                        # Get the parent folder name (e.g., 0001, 0002, ...)
                        parent_folder_name = os.path.basename(root)
                        # Get the current folder name (e.g., 0, 1, 2, ...)
                        current_folder_name = os.path.basename(folder_path)
                        # New filename format: parent_folder_current_folder_oldname.extension
                        new_filename = f"{parent_folder_name}_{current_folder_name}_{file_name}{file_ext}"
                        # New file path
                        new_file_path = os.path.join(folder_path, new_filename)
                        # Rename the file
                        os.rename(file_path, new_file_path)
                        
def merge_all_file_folders(top_folder_path):
    """Flatten the wavelet GUI input tree into a single ``all_files`` folder.

    Creates ``<top_folder_path>/all_files`` (if missing), moves
    every regular file under ``top_folder_path`` into it, and
    bottom-up removes any directory that becomes empty as a
    result.

    Parameters
    ----------
    top_folder_path : str
        Top-level wavelet prosody data directory.

    Returns
    -------
    None
    """
    # Create all_files folder
    all_files_folder = os.path.join(top_folder_path, "all_files")
    os.makedirs(all_files_folder, exist_ok=True)

    # Move all files to all_files folder
    for root, dirs, files in os.walk(top_folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                # Move the file to all_files folder
                shutil.move(file_path, os.path.join(all_files_folder, filename))

    # Delete empty subfolders
    for root, dirs, files in os.walk(top_folder_path, topdown=False):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            # Only delete if the directory is empty
            if not os.listdir(folder_path):
                os.rmdir(folder_path)

# Call the function to perform the operations
rename_files_and_delete_txt_files(top_folder_path)

# Call the function to perform the operations
merge_all_file_folders(top_folder_path)