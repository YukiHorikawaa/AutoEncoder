import os
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Modeledit():
    def __init__(self, Folder_PATH):
        self.Folder_PATH =  Folder_PATH
        self.now = dt.datetime.now()
        self.filename = os.path.join(self.Folder_PATH, "model_data", self.now.strftime('%Y%m%d'))
        try:
            os.makedirs(self.filename)
        except FileExistsError:
            pass
        # モデル保存
    def save_model(self, model, name):
        modelPath = os.path.join(self.filename, name + ".pth")
        torch.save(model.state_dict(), modelPath)

    def read_model(self, model, path):
        # return model.load_state_dict(torch.load(path), strict=False)
        return model.load_state_dict(torch.load(path))

class Autoencoder(nn.Module):
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

#----------------------BatchNorm----------------------
class BatchNorm(nn.Module):
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(np.ones(shape, dtype='float32')))
        self.beta = nn.Parameter(torch.tensor(np.zeros(shape, dtype='float32')))
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        std = torch.std(x, 1, keepdim=True)
        x_normalized = (x - mean) / (std**2 + self.epsilon)**0.5
        return self.gamma * x_normalized + self.beta
#----------------------BatchNorm----------------------
class Autoencoder2_batchNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        #N(バッチサイズ), 784(ピクセル数64x64)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5),
            # nn.MaxPool1d(kernel_size=5, stride= 3)
            BatchNorm(()),
            nn.ReLU(),
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3),
            BatchNorm(()),
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
            BatchNorm((16,3)),
            nn.ReLU(),
            nn.Unflatten(0, (1, 1 ,16)),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1),
            BatchNorm(()),
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
#ーーーーーーーーーーーーー以下学習がうまく行かなかったーーーーーーーーーーーーーーーーーーーーーーー
class Autoencoder0512(nn.Module):
    def __init__(self):
        super().__init__()
        #N(バッチサイズ), 784(ピクセル数64x64)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5),
            # nn.MaxPool1d(kernel_size=5, stride= 3)
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(16, 12), #N,16 -> N,12
            nn.ReLU(),
            nn.Linear(12, 3), #N,12 -> N,3\
            # nn.Linear(16, 3), #N,16 -> N,12
        )
        self.decoder = nn.Sequential(
            # nn.Linear(3,  16),
            nn.Linear(3, 12), #N,784 -> N,128
            nn.ReLU(),
            nn.Linear(12, 16), 
            nn.ReLU(),
            nn.Unflatten(0, (1, 1 ,16)),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1),
            # nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5, padding = 1, output_padding=1),
            # nn.Dropout(p=0.2),
            # nn.Flatten(0, -1),
            nn.ReLU(),
            # nn.Sigmoid(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #N(バッチサイズ), 784(ピクセル数64x64)
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5),
#             # nn.MaxPool1d(kernel_size=5, stride= 3)
#             nn.ReLU(),
#             nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3),
#             # nn.ReLU(),
#             # nn.Flatten(0, -1),
#             # nn.Linear(16, 12), #N,64 -> N,12
#             # nn.ReLU(),
#             # nn.Linear(12, 3), #N,12 -> N,3
#         )
#         self.decoder = nn.Sequential(
#             # nn.Linear(3, 12), #N,784 -> N,128
#             # nn.ReLU(),
#             # nn.Linear(12, 16), #N,128 -> N,64
#             # nn.ReLU(),
#             # nn.Unflatten(0, (1, 1 ,16)),
#             nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5, padding = 1, output_padding=1),
#             # nn.Flatten(0, -1),
#             # nn.ReLU(),
#             nn.Sigmoid(),
#         )
    
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded