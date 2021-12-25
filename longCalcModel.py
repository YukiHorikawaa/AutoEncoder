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
from LoopModel import loopmodel


global Dataset
Dataset = dataset.dataset(npyFlag=True)
global epochStr
epochStr = 150
global epochSizeStr
epochSizeStr = 2048
global epochList
epochList = np.arange(300, 1001, 100)
global epochSizeList
epochSizeList = [256*i for i in range(1, 5)]
global folder_name
folder_name = "newSensorLoop"


# def modelTrain():
#     losslist=[]
#     #model
#     model_CnnAE = mainmodel.Autoencoder_cnn()#model1
#     model_norm = mainmodel.Autoencoder_btchnorm()#model2
#     #loss
#     criterion_mse = nn.MSELoss()
#     criterion_ssim = LossFunction.SSIMLoss()

#     model = model_norm
#     criterion = criterion_mse

#     #optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

#     outputs = []
#     #テンソル型に変換
#     input = torch.from_numpy(data.astype(np.float64)).clone()
#     ori_input = torch.from_numpy(ori_data.astype(np.float64)).clone()
#     cnt = 0
#     for (epoch, epoch_ori) in zip(input, ori_input):
#         cnt+=1
#         for (x, y) in zip(epoch, epoch_ori):
#             recon = model(x)
#             # print("type:recon_data{}".format(type(recon)))
#             loss = criterion(recon, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # print('epoch:{}, Loss:{:.4f}'.format(cnt, float(loss)))
#         losslist.append(loss)
#         outputs.append((epoch, data, recon))
#     return model

for i in epochList:
    for j in epochSizeList:
        epochStr = int(i)
        epochSizeStr = int(j)
        data, ori_data, test_data , anomaly_data= Dataset.read_noised_traindata("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_train/train.npy", "/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_anomaly/anomaly.npy", epochStr, epochSizeStr, 1, readType=False)
        
        model = loopmodel(data, ori_data, test_data , anomaly_data)

        model_name = "Denoising_AE_CNN"+"epoch_"+str(epochStr)+"epockSize_"+str(epochSizeStr)
        ModelEdit = mainmodel.Modeledit(folder_name,model_name)
        ModelEdit.save_model(model)
        print("{}→{}".format(model_name, datetime.datetime.now()))

