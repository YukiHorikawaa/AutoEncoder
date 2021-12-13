import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("make")
        # 初期化処理
        # self.param = ... 
    def forward(self, recon_data, x, window = 8, alfa=1,beta=1,gamma=1,c1=0.01,c2=0.03):
        print("type:x{},recon_data{}".format(type(x),type(recon_data)))
        # print("data:x{},recon_data{}".format(x,recon_data))
        recon_data =  recon_data.flatten().detach().numpy().astype(np.float64)
        x =  x.flatten().detach().numpy().astype(np.float64)
        input_size = len(x)
        inPos = 0
        outPos = window
        SSIM_val = 0.0
        # M = int(input_size/window)
        M = input_size/window
        M = int(M)
        print("type:x{},recon_data{}".format(type(x),type(recon_data)))
        print("size:x{},recon_data{}".format(x.shape,recon_data.shape))
        for outPos in range(0, input_size+window, window):
            x_batch = x[int(inPos):int(outPos)]
            recon_batch = recon_data[int(inPos):int(outPos)]
            #[in:out]の配列の数がおかしくてエラー
            print("type:x_batch{},x_batch{}".format(type(x_batch),type(x_batch)))
            print("size:x_batch{},x_batch{}".format(x_batch.shape,x_batch.shape))
            x_mean = np.mean(x_batch)
            recon_mean = np.mean(recon_batch)
            x_std = np.std(x_batch)
            recon_std = np.std(recon_batch)
            #covが二次元なのはおかしい
            x_cov = np.cov([x_batch,recon_batch])
            SSIM_val += ((2*x_mean*recon_mean+c1)*(2*x_cov+c2))/((x_mean**2+recon_mean**2+c1)*(x_std**2+recon_std**2+c2))
            outPos = inPos
        out = SSIM_val/M
        print(out)
        print(out.shape)
        out = torch.tensor(int(out), requires_grad=True)
        return out