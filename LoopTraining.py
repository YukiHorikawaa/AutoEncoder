# %% [markdown]
# # Denoising AutoEncoder
# 
# 1. CNN+AE
# 2. CNN+AE+BatchNorm
# 3. CNN+AE+SSIMLoss

# %%
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
import datetime

# %% [markdown]
# ## 学習データ作成
# 
# old Sensr Data

# %%

# Dataset = dataset.dataset("Obrid_AE", "data")
# Dataset.concat_data("sample_data",500)
# Dataset = dataset.dataset("Obrid_AE", "test")
# print("----------------------")
# Dataset.concat_data("sample_test",100)
# print("----------------------")
# data = Dataset.read_savedata("sample_test")
# print(data.shape[0])
# print("----------------------")
# data, ori_data, test_data , anomaly_data= Dataset.read_noised_traindata("sample_data", "sample_test", 1000, 128, 1)


# %% [markdown]
# ## New Sensor Data

# %%


global Dataset
Dataset = dataset.dataset(npyFlag=True)
global epochStr
epochStr = 150
global epochSizeStr
epochSizeStr = 2048
global epochList
epochList = np.arange(100, 1001, 100)
global epochSizeList
epochSizeList = [256*i for i in range(1, 8)]
global folder_name
folder_name = "oldSensorLoop"
epochStr = 300
epochSizeStr = 1024
for i in epochList:
    for j in epochSizeList:
        epochStr = int(i)
        epochSizeStr = int(j)
        # data, ori_data, test_data , anomaly_data= Dataset.read_noised_traindata("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_train/train.npy", "/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_anomaly/anomaly.npy", epochStr, epochSizeStr, 1, readType=False)
        data, ori_data, test_data , anomaly_data= Dataset.read_noised_traindata("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/oldSensor/train/train1226.csv", "/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/oldSensor/anomaly/anomaly1226.csv", epochStr, epochSizeStr, 1, readType=True)

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
        #テンソル型に変換
        input = torch.from_numpy(data.astype(np.float32)).clone()
        ori_input = torch.from_numpy(ori_data.astype(np.float32)).clone()
        cnt = 0
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
        model_name = "Denoising_AE_CNN"+"epoch_"+str(epochStr)+"epockSize_"+str(epochSizeStr)
        ModelEdit = mainmodel.Modeledit(folder_name,model_name)
        ModelEdit.save_model(model)
        print("{}→{}".format(model_name, datetime.datetime.now()))

# %% [markdown]
# ## モデルの保存

# %%
# folder_name = "newSensor"
# model_name = "Denoising_AE_CNN"+"epoch_"+str(epochStr)+"epockSize_"+str(epochSizeStr)
# ModelEdit = mainmodel.Modeledit(folder_name,model_name)
# ModelEdit.save_model(model)

        


# %% [markdown]
# ## モデル学習
# 
# Denoise AE
# 

# %%
