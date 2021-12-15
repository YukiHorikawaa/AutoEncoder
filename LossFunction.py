from numpy.lib import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

def ori_cov(x,y,x_mean,y_mean,data_len):
    hensa = 0
    # print(x)
    # print(y)
    for i in range(data_len):
        hensa += (x[i]-x_mean)*(y[i]-y_mean)
    return hensa/data_len



class SSIMLoss(nn.Module):
    """
    画像処理で用いられる評価式を用いたLoss関数
    """
    def __init__(self):
        super().__init__()
        print("make")
        # 初期化処理
        # self.param = ... 
    def forward(self, recon_data, x, window = 8, alfa=1,beta=1,gamma=1,c1=0.01,c2=0.03):
        # print("type:x{},recon_data{}".format(type(x),type(recon_data)))
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
        # print("type:x{},recon_data{}".format(type(x),type(recon_data)))
        # print("size:x{},recon_data{}".format(x.shape,recon_data.shape))
        for outPos in range(0, input_size+window, window):
            try:
                x_batch = x[int(inPos):int(outPos)]
                recon_batch = recon_data[int(inPos):int(outPos)]
                #[in:out]の配列の数がおかしくてエラー
                # print("type:x_batch{},x_batch{}".format(type(x_batch),type(recon_batch)))
                # print("size:x_batch{},x_batch{}".format(x_batch.shape,recon_batch.shape))
                # print("size:x{},x{}".format(x_batch,recon_batch))
                x_mean = np.mean(x_batch)
                recon_mean = np.mean(recon_batch)
                x_std = np.std(x_batch)
                recon_std = np.std(recon_batch)
                #covが二次元なのはおかしい
                # x_cov = np.cov([x_batch,recon_batch])
                x_cov = ori_cov(x_batch,recon_batch, x_mean, recon_mean, window)
                # x_cov = x_cov[0,0]**2+x_cov[1,1]**2+x_cov[0,1]**2+x_cov[1,0]**2
                # print("x_mean:{}r_mean:{}x_std:{}r_std:{}x_cov:{}".format(x_mean,recon_mean,x_std,recon_std,x_cov))
                # SSIM_val += ((2*x_mean*recon_mean+c1)*(2*x_cov+c2))/((x_mean**2+recon_mean**2+c1)*(x_std**2+recon_std**2+c2))
                #共分散が発散してしまい、nanになるため
                # print("(2*x_mean*recon_mean+c1)",(2*x_mean*recon_mean+c1))
                # print("(x_mean**2+recon_mean**2+c1)",(x_mean**2+recon_mean**2+c1))
                # print("(x_std**2+recon_std**2+c2)",(x_std**2+recon_std**2+c2))
                # print("((x_mean**2+recon_mean**2+c1)*(x_std**2+recon_std**2+c2))",((x_mean**2+recon_mean**2+c1)*(x_std**2+recon_std**2+c2)))
                # loss_data = (2*x_mean*recon_mean+c1)/((x_mean**2+recon_mean**2+c1)*(x_std**2+recon_std**2+c2))
                loss_data = ((2*x_mean*recon_mean+c1)*(2*x_cov+c2))/((x_mean**2+recon_mean**2+c1)*(x_std**2+recon_std**2+c2))#論文通りの式
                # print("loss_data",loss_data)
                if math.isnan(loss_data):
                    # print("SSIM_val","error")
                    pass
                else:
                    SSIM_val += loss_data
                    # print("SSIM_val",SSIM_val)
                inPos = outPos
            except IndexError: 
                pass
        out = SSIM_val/M
        # print(out)
        out = torch.tensor(out, requires_grad=True)
        return out

class SSIMLoss_edit(nn.Module):
    def __init__(self):
        super().__init__()
        print("make")
        # 初期化処理
        # self.param = ... 
    def forward(self, recon_data, x, window = 4, alfa=1,beta=1,gamma=1,c1=0.01,c2=0.03):
        # print("type:x{},recon_data{}".format(type(x),type(recon_data)))
        # print("data:x{},recon_data{}".format(x,recon_data))
        recon_data =  recon_data.flatten().detach().numpy().astype(np.float64)
        x =  x.flatten().detach().numpy().astype(np.float64)
        input_size = len(x)
        inPos = 0
        outPos = window
        SSIM_val = 0.0
        M = input_size/window
        M = int(M)
        for outPos in range(0, input_size+window, window):
            x_batch = x[int(inPos):int(outPos)]
            recon_batch = recon_data[int(inPos):int(outPos)]
            x_mean = np.mean(x_batch)
            recon_mean = np.mean(recon_batch)
            x_std = np.std(x_batch)
            recon_std = np.std(recon_batch)
            loss_data = x_mean-recon_mean+x_std-recon_std
            if math.isnan(loss_data):
                # print("SSIM_val","error")
                pass
            else:
                SSIM_val += loss_data
                # print("SSIM_val",SSIM_val)
            inPos = outPos
        out = SSIM_val/M
        # print(out)
        out = torch.tensor(out, requires_grad=True)
        return out