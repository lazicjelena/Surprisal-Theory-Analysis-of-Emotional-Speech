# -*- coding: utf-8 -*-
"""resampling.py

Jelenin fajl
lazic.jelenaa@gmail.com

Scripta vrsi promjena ucestanosti odabiranja jednog audio signala.
Samo jedan snimak ima ucestanost odabiranja duplo vecu nego svi ostali snimci.
Da bi se moglo iyvrsiti transkriptovanje snimka potrebno je prvo promjeniti ucestanost
odabiranja i snimiti ga sa novom ucestanoscu.

Pipeline role
-------------
One-off audio preprocessing step that runs BEFORE forced alignment
(``Forced alignment/novosadska_baza_podataka.py``). One speaker in the
corpus (id 1052) was recorded at twice the corpus-wide sampling rate;
this script walks that speaker's recording folder and writes a
sample-rate-uniform copy under a sibling ``1052_Resampled`` folder,
preserving the per-emotion subfolder structure (0..4). Only after this
script runs are all WAV files in the corpus uniformly sampled and ready
for the Google Cloud Speech-to-Text pipeline.

"""

import os
from pydub import AudioSegment

def resample_wav(input_file, output_file, target_sampling_rate):
    """Read a WAV file, change its sampling rate, and save the result.

    Parameters
    ----------
    input_file : str
        Path to the source WAV file.
    output_file : str
        Path where the resampled WAV file will be written.
    target_sampling_rate : int
        Desired sampling frequency in Hz (e.g. 44100).

    Returns
    -------
    None

    Side effects
    ------------
    Writes a WAV file at ``output_file``. Overwrites if it already exists.
    """
    audio = AudioSegment.from_wav(input_file)
    resampled_audio = audio.set_frame_rate(target_sampling_rate)
    resampled_audio.export(output_file, format="wav")

def resample_files_in_directory(root_folder, target_sampling_rate):
    """Recursively resample every ``.wav`` inside ``root_folder``.

    Walks ``root_folder`` with ``os.walk``, finds every ``.wav`` file, and
    writes a resampled copy into a sibling ``1052_Resampled`` directory.
    The per-emotion subfolder structure (0..4) of the source is mirrored
    in the output. The output root is derived by stripping the last 5
    characters of ``root_folder`` and appending ``1052_Resampled``.

    Parameters
    ----------
    root_folder : str
        Folder containing the speaker's WAV files (one subfolder per
        emotion). The last 5 characters of this path are stripped to
        compute the output root.
    target_sampling_rate : int
        Desired sampling frequency in Hz.

    Returns
    -------
    None

    Side effects
    ------------
    Creates ``<parent>/1052_Resampled/<emotion>/`` folders if missing
    and writes one resampled WAV file for each input WAV.
    """
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