import os
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Autoencoder2(nn.Module):
    def __init__(self):
        super().__init__()
        #N(バッチサイズ), 784(ピクセル数64x64)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5),
            # nn.MaxPool1d(kernel_size=5, stride= 3)
            nn.ReLU(),
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3),
            nn.ReLU(),
            nn.Flatten(0, -1),
            # nn.Linear(16, 12), #N,16 -> N,12
            # nn.ReLU(),
            # nn.Linear(12, 3), #N,12 -> N,3\
            nn.Linear(16, 3), #N,16 -> N,12
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,  16),
            # nn.Linear(3, 12), #N,784 -> N,128
            # nn.ReLU(),
            # nn.Linear(12, 16), 
            nn.ReLU(),
            nn.Unflatten(0, (1, 1 ,16)),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5, padding = 1, output_padding=1),
            # nn.Flatten(0, -1),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
