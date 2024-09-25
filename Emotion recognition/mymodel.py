# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:05:31 2024

@author: Jelena

Model Architecture
"""

import torch.nn as nn

# make model
class LFLB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=4, pool_stride=4):
        super(LFLB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(pool_size, pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.LFLB_1 = LFLB(in_channels=1, out_channels=64, pool_size=2, pool_stride=2)
        self.LFLB_2 = LFLB(in_channels=64, out_channels=64)
        self.LFLB_3 = LFLB(in_channels=64, out_channels=128)
        self.LFLB_4 = LFLB(in_channels=128, out_channels=128)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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