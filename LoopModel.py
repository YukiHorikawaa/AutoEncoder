import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
import dataset
import mainmodel
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve
import modelVAE
from LossFunction import SSIMLoss #自作Loss関数
import LossFunction
import numpy as np
import datetime


#------------学習----------
def loopmodel(data, ori_data, test_data , anomaly_data):
    losslist=[]
    #model
    model_CnnAE = mainmodel.Autoencoder_cnn()#model1
    model_norm = mainmodel.Autoencoder_batchnorm()#model2
    #loss
    criterion_mse = nn.MSELoss()
    criterion_ssim = LossFunction.SSIMLoss()

    model = model_norm
    criterion = criterion_mse

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

    outputs = []
    loss = 0
    epoch = 0
    data = 0
    recon = 0
    cnt = 0
    #テンソル型に変換
    input = torch.from_numpy(data.astype(np.float64)).clone()
    ori_input = torch.from_numpy(ori_data.astype(np.float64)).clone()
    for (epoch, epoch_ori) in zip(input, ori_input):
        cnt+=1
        for (x, y) in zip(epoch, epoch_ori):
            recon = model(x)
            # print("type:recon_data{}".format(type(recon)))
            loss = criterion(recon, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('epoch:{}, Loss:{:.4f}'.format(cnt, float(loss)))
        losslist.append(loss)
        outputs.append((epoch, data, recon))
    #------------学習----------
    return model
