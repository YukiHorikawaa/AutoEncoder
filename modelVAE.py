import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))


class VAE_dropout(nn.Module):
    """
    Dropout層
    """
    def __init__(self, z_dim, device):
        super(VAE_dropout, self).__init__()
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

    # torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))


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

class VAE_linear(nn.Module):
    """
    DLkiso参考
    全結合層のみを用いたモデル
    """
    def __init__(self, z_dim, device):
        super(VAE_linear, self).__init__()
        self.flat = nn.Flatten(0, -1)
        self.enc1 = nn.Linear(256, 128) #N,256 -> N,128
        self.enc2 = nn.Linear(128, 64) #N,128 -> N,64
        self.enc3 = nn.Linear(64, 12) #N,64 -> N,12
        self.enc4 = nn.Linear(12, z_dim)

        self.encmean = nn.Linear(12, z_dim)
        self.encvar = nn.Linear(12, z_dim)

        # self.dec1 = nn.Linear(z_dim, 200)
        self.dec1 = nn.Linear(z_dim,  12)
        self.dec2 = nn.Linear(12, 64) #N,12 -> N,64
        self.dec3 = nn.Linear(64, 128) #N,64 -> N,128
        self.dec4 = nn.Linear(128, 256) #N,128 -> N,256
        self.unflat = nn.Unflatten(0, (1, 1 ,256))
        self.device = device


    def _encoder(self, x):
        x = self.flat(x)
        # print(x.shape)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        # x = F.relu(self.enc4(x))
        mean = self.encmean(x)
        var = F.softplus(self.encvar(x))
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = self.unflat(x)
        x = torch.sigmoid(x)
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


class VAE_cnn_net(nn.Module):
    """
    CNN層を用いたVAE
    https://deepblue-ts.co.jp/image-generation/pytorch_vae/
    """
    def __init__(self, z_dim, device):
        super(VAE_cnn_net, self).__init__()
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
        var = self.encvar(x)
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        # return mean + torch.sqrt(var) * epsilon
        return mean + epsilon*torch.exp(0.5 * var)

    def _decoder(self, z):
        x = F.relu(self.dec1(z))
        x = self.unflat(x)
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

    def forward(self, x, device):
        mean, var = self._encoder(x)
        # print(mean, var)
        # KL lossの計算
        # KL = -0.5 * torch.mean(torch.sum(1 + torch_log(var) - mean**2 - var, dim=0))
        KL = 0.5 * torch.sum(1+var - mean**2 - torch.exp(var)) # KL[q(z|x)||p(z)]を計算
        
        z = self._sample_z(mean, var)
        y = self._decoder(z)

        # reconstruction lossの計算
        # reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1))
        reconstruction = torch.sum(x * torch.log(y+1e-8) + (1 - x) * torch.log(1 - y  + 1e-8)) #E[log p(x|z)]
        lower_bound = -(KL + reconstruction)

        return lower_bound, z, y

class VAE_cnn_drop_net(nn.Module):
    """
    CNN層を用いたVAE
    https://deepblue-ts.co.jp/image-generation/pytorch_vae/
    """
    def __init__(self, z_dim, device):
        super(VAE_cnn_drop_net, self).__init__()
        self.device = device
        self.enc1 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5)
        self.enc1_drop = nn.Dropout(p=0.2)  # [new] Dropoutを追加してみる
        self.enc2 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3)
        self.enc2_drop = nn.Dropout(p=0.2)  # [new] Dropoutを追加してみる
        self.flat = nn.Flatten(0, -1)
        self.enc3 = nn.Linear(16, z_dim)

        self.encmean = nn.Linear(16, z_dim)
        self.encvar = nn.Linear(16, z_dim)

        # self.dec1 = nn.Linear(z_dim, 200)
        self.dec1 = nn.Linear(z_dim,  16)
        self.unflat = nn.Unflatten(0, (1, 1 ,16))
        self.dec1_drop = nn.Dropout(p=0.2)  # [new] Dropoutを追加してみる
        self.dec2 = nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3, output_padding=1)
        self.dec2_drop = nn.Dropout(p=0.2)  # [new] Dropoutを追加してみる
        self.dec3 = nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 5, padding = 1, output_padding=1)


    def _encoder(self, x):
        x = F.relu(self.enc1(x))
        x = self.enc1_drop(x)
        x = F.relu(self.enc2(x))
        x = self.enc2_drop(x)
        x = self.flat(x)
        mean = self.encmean(x)
        var = self.encvar(x)
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        # return mean + torch.sqrt(var) * epsilon
        return mean + epsilon*torch.exp(0.5 * var)

    def _decoder(self, z):
        x = F.relu(self.dec1(z))
        x = self.unflat(x)
        x = self.dec1_drop(x)
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

    def forward(self, x, device):
        mean, var = self._encoder(x)
        # print(mean, var)
        # KL lossの計算
        # KL = -0.5 * torch.mean(torch.sum(1 + torch_log(var) - mean**2 - var, dim=0))
        KL = 0.5 * torch.sum(1+var - mean**2 - torch.exp(var)) # KL[q(z|x)||p(z)]を計算
        
        z = self._sample_z(mean, var)
        y = self._decoder(z)

        # reconstruction lossの計算
        # reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1))
        reconstruction = torch.sum(x * torch.log(y+1e-8) + (1 - x) * torch.log(1 - y  + 1e-8)) #E[log p(x|z)]
        lower_bound = -(KL + reconstruction)

        return lower_bound, z, y

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        #N(バッチサイズ), 784(ピクセル数64x64)
        self.encoder = nn.Sequential(
            nn.Linear(256, 128), #N,256 -> N,128
            nn.ReLU(),
            nn.Linear(128, 64), #N,128 -> N,64
            nn.ReLU(),
            nn.Linear(64, 12), #N,64 -> N,12
            nn.ReLU(),
            nn.Linear(12, 3), #N,12 -> N,3
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), #N,3 -> N,12
            nn.ReLU(),
            nn.Linear(12, 64), #N,12 -> N,64
            nn.ReLU(),
            nn.Linear(64, 128), #N,64 -> N,128
            nn.ReLU(),
            nn.Linear(128, 256), #N,128 -> N,256
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded