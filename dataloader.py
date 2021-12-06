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
        input = input[np.newaxis, np.newaxis, :]
        recon = model(input).detach().numpy()
        encoded = model.encoder(input)

        input_list.append(input)
        recon_list.append(recon)
        encoded_list.append(encoded)
        try:
            if ((dataLen/i+1)*100) % 10 == 0:
                print("--------{}/{}--------".format(i,dataLen))
        except ZeroDivisionError:
            pass

    return recon_list, encoded_list, input_list
