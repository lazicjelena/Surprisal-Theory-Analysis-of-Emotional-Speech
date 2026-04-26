# -*- coding: utf-8 -*-
"""audiodataset.py

Created on Tue Sep 24 08:18:26 2024

@author: Jelena

Pipeline role
-------------
Audio data feeding for the ``Emotion recognition/`` model.
Defines :class:`AudioDataset`, a ``torch.utils.data.Dataset``
that pulls a ``(file_path, last_folder)`` pair out of the per-row
DataFrame produced by ``create_datasets.py``, loads the WAV via
:func:`librosa.load` at 44.1 kHz and applies the shared
:func:`audio_utils.get_fixed_length_mel_spectrogram` (80 mel
bands x 250 frames) to obtain the model input. The
:func:`create_dataloader` helper wraps the dataset in a
:class:`torch.utils.data.DataLoader` with shuffling enabled.
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torch

from emotion_recognition.audio_utils import get_fixed_length_mel_spectrogram


mel_dim = 80

class AudioDataset(Dataset):
    """``torch.utils.data.Dataset`` over the per-utterance DataFrame.

    Wraps the per-row CSVs produced by ``create_datasets.py``
    (or ``create_csv_for_testing_synthetisized_data.py``) and
    serves ``(mel_spectrogram, label)`` tensor pairs ready for the
    ``MyModel`` classifier.
    """

    def __init__(self, dataframe, max_length=250):
        """Store the wrapped DataFrame and the target spectrogram length.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Two-column table whose first column is a WAV file path
            and second column is the integer emotion label.
        max_length : int, optional
            Number of time frames to which every Mel spectrogram is
            padded or truncated. Defaults to ``250``.
        """
        self.dataframe = dataframe
        self.max_length = max_length

    def __len__(self):
        """Return the number of utterances in the wrapped DataFrame."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Return the ``(mel_tensor, label_tensor)`` pair for row ``idx``.

        The WAV at ``self.dataframe.iloc[idx, 0]`` is loaded at
        44.1 kHz and converted to a fixed-length log-Mel
        spectrogram via :meth:`_get_mel_spectrogram`. A leading
        channel axis is added so the output is ready for a 2-D CNN
        (``(1, n_mels, max_length)``).

        Parameters
        ----------
        idx : int
            Row index into the wrapped DataFrame.

        Returns
        -------
        tuple of torch.Tensor
            ``(mel_spectrogram, label)`` where ``mel_spectrogram``
            has shape ``(1, 80, max_length)`` and ``label`` is a
            scalar ``torch.long``.
        """
        file_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        
        # Load the audio file using librosa to apply mel spectrogram
        waveform, sr = librosa.load(file_path, sr=44100)
        
        # Compute the mel spectrogram
        mel_spectrogram = self._get_mel_spectrogram(waveform, sr)
        
        return torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0) , torch.tensor(label, dtype=torch.long)

    def _get_mel_spectrogram(self, waveform, sr):
        """Thin wrapper around :func:`audio_utils.get_fixed_length_mel_spectrogram`.

        Uses the module-level ``mel_dim`` constant for the number
        of Mel bands and ``self.max_length`` for the time-axis
        size.

        Parameters
        ----------
        waveform : numpy.ndarray
            Mono waveform.
        sr : int
            Sampling rate in Hz.

        Returns
        -------
        numpy.ndarray
            Log-Mel spectrogram of shape ``(mel_dim, max_length)``.
        """
        # Extract Mel spectrogram using the predefined function
        mel_spectrogram = get_fixed_length_mel_spectrogram(waveform, sr, n_mels=mel_dim, fixed_length=self.max_length)
        return mel_spectrogram
    
    
# Example function to create DataLoader (not part of the training loop yet)
def create_dataloader(dataframe, batch_size):
    """Wrap ``dataframe`` in an :class:`AudioDataset` + shuffling DataLoader.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Two-column ``(file_path, label)`` table.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    torch.utils.data.DataLoader
        Shuffling DataLoader yielding ``(mel_tensor, label)``
        batches.
    """
    dataset = AudioDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader