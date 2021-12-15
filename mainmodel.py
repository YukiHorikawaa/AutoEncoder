import os
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Modeledit():
    def __init__(self, Folder_PATH, model_name):
        self.Folder_PATH =  Folder_PATH
        self.model_name = model_name
        self.now = dt.datetime.now()
        self.filename = os.path.join(self.Folder_PATH, self.model_name, self.now.strftime('%Y%m%d'))
        try:
            os.makedirs(self.filename)
        except FileExistsError:
            pass
        # モデル保存
    def save_model(self, model):
        modelPath = os.path.join(self.filename, self.model_name + self.now.strftime('%H%M%S') + ".pth")
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
    
class Autoencoder_cnn(nn.Module):
    """
    CNN+AE
    """
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


class Autoencoder_batchnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5)
        self.enc2 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3)
        self.flat = nn.Flatten(0, -1)
        self.enc3 = nn.Linear(16, 3)

        self.dec1 = nn.Linear(3,  16)
        self.unflat = nn.Unflatten(0, (1, 1 ,16))
        self.dec2 = nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5, padding = 1, output_padding=1)

    def encoder(self, x):
        x = F.relu(self.enc1(x))
        BatchNorm((1, 1, 50))
        # print(x.shape)
        x = F.relu(self.enc2(x))
        BatchNorm((1, 1, 16))
        # print(x.shape)
        x = self.flat(x)
        x = self.enc3(x)
        return x
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        BatchNorm((16))
        # print(x.shape)
        x = self.unflat(x)
        x = F.relu(self.dec2(x))
        BatchNorm((1, 1, 51))
        # print(x.shape)
        x = torch.sigmoid(self.dec3(x))
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAE_cnn(nn.Module):
    """
    CNN層を用いたVAE
    """
    def __init__(self, z_dim, device):
        super(VAE_cnn, self).__init__()
        self.device = device
        self.enc1 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5)
        self.enc2 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3)
        self.flat = nn.Flatten(0, -1)
        self.enc3 = nn.Linear(16, z_dim)

        self.encmean = nn.Linear(16, z_dim)
        self.encvar = nn.Linear(16, z_dim)

        # self.dec1 = nn.Linear(z_dim, 200)
        self.dec1 = nn.Linear(z_dim,  16)
        self.unflat = nn.Unflatten(0, (1, 1 ,16))
        self.dec2 = nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5, padding = 1, output_padding=1)


    def _encoder(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.flat(x)
        mean = self.encmean(x)
        var = F.softplus(self.encvar(x))
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
        x = F.relu(self.dec1(z))
        x = self.unflat(x)
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z

    def loss(self, x):
        mean, var = self._encoder(x)
        # print(mean, var)
        # KL lossの計算
        KL = -0.5 * torch.mean(torch.sum(1 + torch_log(var) - mean**2 - var, dim=0))
        
        z = self._sample_z(mean, var)
        y = self._decoder(z)

        # reconstruction lossの計算
        reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1))

        return KL, -reconstruction 

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