import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
import dataset
import mainmodel
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve

def ocsvm_dataset(model, data):
    input_list = []
    recon_list = []
    encoded_list = []
    dataLen = data.shape[0]
    for i in range(dataLen):
        input =  torch.from_numpy((data[i]).astype(np.float32)).clone()
        input_list.append(input)
        input = input[np.newaxis, np.newaxis, :]
        recon = model(input).detach().numpy()
        encoded = model.encoder(input).detach().numpy()
        recon_list.append(recon)
        encoded_list.append(encoded)
        try:
            if ((dataLen/i+1)*100) % 10 == 0:
                print("--------{}/{}--------".format(i,dataLen))
        except ZeroDivisionError:
            pass

    input_list = np.array(input_list)
    recon_list = np.array(recon_list)
    recon_list = np.squeeze(recon_list,1)
    recon_list = np.squeeze(recon_list,1)
    encoded_list = np.array(encoded_list)

    print("recon_list:{}".format(recon_list.shape))
    print("encoded_list:{}".format(encoded_list.shape))
    print("input_list:{}".format(input_list.shape))

    return recon_list, encoded_list, input_list
def ocsvm_dataset_VAEmodel(model, device, data):
    input_list = []
    recon_list = []
    encoded_list = []
    dataLen = data.shape[0]
    for i in range(dataLen):
        input =  torch.from_numpy((data[i]).astype(np.float32)).clone()
        input_list.append(input)
        input = input[np.newaxis, np.newaxis, :]
        loss, z, recon = model(input, device)
        # encoded = model.encoder(input).detach().numpy()
        recon_list.append(recon.detach().numpy().flatten())
        encoded_list.append(z.detach().numpy())
        try:
            if ((dataLen/i+1)*100) % 10 == 0:
                print("--------{}/{}--------".format(i,dataLen))
        except ZeroDivisionError:
            pass

    input_list = np.array(input_list)
    recon_list = np.array(recon_list)
    # recon_list = np.squeeze(recon_list,1)
    # recon_list = np.squeeze(recon_list,1)
    encoded_list = np.array(encoded_list)

    print("recon_list:{}".format(recon_list.shape))
    print("encoded_list:{}".format(encoded_list.shape))
    print("input_list:{}".format(input_list.shape))

    return recon_list, encoded_list, input_list
def ocsvm_DAE_dataset(data):
    list = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            list.append(data[i, j, :,:,:].flatten())
    return np.array(list)
