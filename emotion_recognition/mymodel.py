# -*- coding: utf-8 -*-
"""mymodel.py

Created on Tue Sep 24 08:05:31 2024

@author: Jelena

Model Architecture

Pipeline role
-------------
Defines the emotion-recognition model used by the
``Emotion recognition/`` training and testing scripts.
:class:`LFLB` (Local Feature-Learning Block) is a
Conv2D + BatchNorm + ELU + MaxPool unit; :class:`MyModel`
stacks four ``LFLB`` blocks (the 4th is currently disabled in
:meth:`MyModel.forward`), reshapes the resulting feature map
into a length-14 sequence of 128-dim feature vectors, runs it
through a single-layer LSTM with 256 hidden units and projects
its last time step through a fully-connected + softmax head to
``num_classes`` emotion logits.
"""

import torch.nn as nn

# make model
class LFLB(nn.Module):
    """Local Feature-Learning Block: Conv2D + BatchNorm + ELU + MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=4, pool_stride=4):
        """Build the four sub-layers of an LFLB block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Convolution kernel size (square). Defaults to ``3``.
        pool_size : int, optional
            Max-pool kernel size (square). Defaults to ``4``.
        pool_stride : int, optional
            Max-pool stride. Defaults to ``4``.
        """
        super(LFLB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(pool_size, pool_stride)

    def forward(self, x):
        """Apply ``conv -> bn -> elu -> pool`` in sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape
            ``(batch, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return x

class MyModel(nn.Module):
    """Emotion-recognition CNN+LSTM classifier on Mel spectrograms."""

    def __init__(self, num_classes):
        """Build the four LFLB blocks, the LSTM and the FC + softmax head.

        Parameters
        ----------
        num_classes : int
            Number of emotion classes.
        """
        super(MyModel, self).__init__()
        self.LFLB_1 = LFLB(in_channels=1, out_channels=64, pool_size=2, pool_stride=2)
        self.LFLB_2 = LFLB(in_channels=64, out_channels=64)
        self.LFLB_3 = LFLB(in_channels=64, out_channels=128)
        self.LFLB_4 = LFLB(in_channels=128, out_channels=128)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Score a batch of Mel spectrograms.

        Note that ``LFLB_4`` is currently bypassed (the call is
        commented out) and the post-LFLB feature map is reshaped
        into a length-14 sequence of 128-dim vectors before the
        LSTM.

        Parameters
        ----------
        x : torch.Tensor
            Input Mel spectrograms of shape
            ``(batch, 1, n_mels, T)``.

        Returns
        -------
        torch.Tensor
            Per-class softmax probabilities of shape
            ``(batch, num_classes)``.
        """
        x = self.LFLB_1(x)
        x = self.LFLB_2(x)
        x = self.LFLB_3(x)
        #x = self.LFLB_4(x)
        
        x = x.reshape(x.shape[0], 2 * 7, 128) 
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use the last output of LSTM
        x = self.fc(x)
        x = self.softmax(x)
        return x