import os
import datetime as dt
import glob #特定のディレクトリに存在するファイルに処理を加える
import csv
import pandas as pd
import numpy as np
import random
class dataset():
    def __init__(self, folder_path, file_name):
        self.Folder_PATH = folder_path
        self.DATA_PATH = os.path.join(self.Folder_PATH, file_name, "*/*.csv")
        self.All_Files = glob.glob(self.DATA_PATH)
        self.test_data_path = None
        try:
            os.makedirs(os.path.join(self.Folder_PATH, "test_data"))
        except FileExistsError:
            pass
        # self.test_data_path = os.path.join(self.Folder_PATH, "test_data", "0503.csv")

    def concat_data(self, outFile_name,shape):
        list=[]
        for path in self.All_Files:
            data_df = pd.read_csv(path, header=None, engine="python")
            data = data_df.iloc[3:, 3].reset_index(drop = True).values
            list.append(data)
        frame=pd.DataFrame(list)
        frame = frame.head(shape)
        out = outFile_name+".csv"
        self.test_data_path = os.path.join(self.Folder_PATH, "test_data", out)
        frame.to_csv(self.test_data_path, index=False, encoding="utf-8")  
        data = self.read_savedata(outFile_name)
        print(data.shape)

    def get_data(self, out_file):
        out = out_file+".csv"
        data_path = os.path.join(self.Folder_PATH, "test_data", out)
        data_df = pd.read_csv(data_path, engine="python")
        return data_df.values

    def read_savedata(self, out_file):
        csv_pass = os.path.join(self.Folder_PATH, "test_data", out_file + ".csv")
        data_df = pd.read_csv(csv_pass, engine="python")
        data = data_df.values
        print(data.shape)
        return data

    def read_traindata(self, out_file, Anomaly_file, epoch_num, epoch_size, ratenum):
        #outfile : 読み込むCSVファイル名
        #epoch_num : エポック数
        #epoch_size : エポックサイズ    
        #ratenum : 学習データとテストデータの割合（ここで指定するのはテストデータの割合）
        data = self.read_savedata(out_file)
        anomaly_data = self.read_savedata(Anomaly_file)
        print(data.max(), data.min())
        #前処理正規化
        data = self.preprocessing(data)
        anomaly_data = self.preprocessing(anomaly_data)
        #学習データとテストデータの分割
        rate = 1 - (ratenum/10)
        print("rate", rate)
        RateData = int(data.shape[0] * rate)
        print("data.shape[0]:", data.shape[0])
        print("rate", RateData)
        
        train_data = data[:RateData]
        test = data[RateData:]

        list=[]
        for i in range(epoch_num):
            #ランダムに配列の番号をランダムに指定
            make_epoch = np.random.randint(0, len(train_data), (epoch_size))
            #ランダムに指定した番号のデータを選択、リストに追加
            list.append(train_data[make_epoch, :])
        train_data = np.array(list)
        # conv1を適用するために３次元
        train_data = train_data[:, :, np.newaxis, np.newaxis, :]
        # test_data = np.array(list)/1024
        print("TrainData", train_data.shape)
        print("TestData", test.shape)
        print("ÄnomalyDta", anomaly_data.shape)
        #学習用データ、テスト用データ、　異常検知のためのテストデータ
        return train_data, test, anomaly_data

    def Augmentation_noise(self, data, rate=0.1):
        data = np.array(data)
        data_max = np.max(data)
        num = data.shape[0]
        randdata = [random.uniform(0, data_max*rate) for i in range(num)]
        # out = data + np.random.rand(0, data_max*rate, num)
        out = data + randdata
        return out
    def Augmentation_plus(self, data, rate=0.1, plus = 0.1):
        data = np.array(data)
        data_max = np.max(data)
        num = data.shape[0]
        randdata = [random.uniform(0, data_max*rate) for i in range(num)]
        # out = data + np.random.rand(0, data_max*rate, num)
        out = data + randdata + plus
        return out
    def Augmentation_minus(self, data, rate=0.1, minus = 0.1):
        data = np.array(data)
        data_max = np.max(data)
        num = data.shape[0]
        randdata = [random.uniform(0, data_max*rate) for i in range(num)]
        # out = data + np.random.rand(0, data_max*rate, num)
        out = data + randdata - minus
        return out
    def read_Auged_traindata(self, out_file, Anomaly_file, epoch_num, epoch_size, ratenum):
        #outfile : 読み込むCSVファイル名
        #epoch_num : エポック数
        #epoch_size : エポックサイズ    
        #ratenum : 学習データとテストデータの割合（ここで指定するのはテストデータの割合）
        data = self.read_savedata(out_file)
        anomaly_data = self.read_savedata(Anomaly_file)
        print(data.max(), data.min())
        #前処理正規化
        data = self.preprocessing(data)
        anomaly_data = self.preprocessing(anomaly_data)
        #学習データとテストデータの分割
        rate = 1 - (ratenum/10)
        print("rate", rate)
        RateData = int(data.shape[0] * rate)
        print("data.shape[0]:", data.shape[0])
        print("rate", RateData)
        
        train_data = data[:RateData]
        test = data[RateData:]
        #Augmentation
        list=[]
        for i in range(train_data.shape[0]):
            list.append(self.Augmentation_noise(train_data[i]))
            list.append(self.Augmentation_plus(train_data[i], 0.1))
            list.append(self.Augmentation_minus(train_data[i], 0.1))
            list.append(self.Augmentation_plus(train_data[i], 0.15))
            list.append(self.Augmentation_minus(train_data[i], 0.15))
            list.append(self.Augmentation_plus(train_data[i], 0.2))
            list.append(self.Augmentation_minus(train_data[i], 0.2))
            list.append(self.Augmentation_noise(train_data[i], 0.2))
            list.append(self.Augmentation_plus(train_data[i], 0.2, 0.1))
            list.append(self.Augmentation_minus(train_data[i], 0.2, 0.1))
            list.append(self.Augmentation_plus(train_data[i], 0.2, 0.15))
            list.append(self.Augmentation_minus(train_data[i], 0.2, 0.15))
            list.append(self.Augmentation_plus(train_data[i], 0.2, 0.2))
            list.append(self.Augmentation_minus(train_data[i], 0.2, 0.2))
        train_data = np.array(list)

        list=[]
        for i in range(epoch_num):
            #ランダムに配列の番号をランダムに指定
            make_epoch = np.random.randint(0, len(train_data), (epoch_size))
            #ランダムに指定した番号のデータを選択、リストに追加
            list.append(train_data[make_epoch, :])
        train_data = np.array(list)
        # conv1を適用するために３次元
        train_data = train_data[:, :, np.newaxis, np.newaxis, :]
        # test_data = np.array(list)/1024
        print("TrainData", train_data.shape)
        print("TestData", test.shape)
        print("ÄnomalyDta", anomaly_data.shape)
        #学習用データ、テスト用データ、　異常検知のためのテストデータ
        return train_data, test, anomaly_data

    def read_train(self, out_file):
        #outfile : 読み込むCSVファイル名
        #epoch_num : エポック数
        #epoch_size : エポックサイズ    
        #ratenum : 学習データとテストデータの割合（ここで指定するのはテストデータの割合）
        data = self.read_savedata(out_file)
        print(data.max(), data.min())
        #前処理正規化
        data = self.preprocessing(data)
        return data
    

    #前処理
    def preprocessing(self, input):
        #今回は正規化
        input = (input - input.min()) / (input.max() - input.min())
        return input

    def savenumpy(self, train, test, anomaly, name):
        now = dt.datetime.now()
        filename = os.path.join(self.Folder_PATH, "test_data", now.strftime('%Y%m%d'))
        try:
            os.makedirs(filename)
        except FileExistsError:
            pass
        np.save(filename + "train"+ str(name), train)
        np.save(filename + "test"+ str(name), test)
        np.save(filename + "anomaly"+ str(name), anomaly)
    def loadnumpy(self,date,name):
        #dateは保存した日付名,nameは保存名
        filename = os.path.join(self.Folder_PATH, "test_data", str(date))
        train=np.load(filename+"train"+str(name)+'.npy')
        test=np.load(filename+"test"+str(name)+'.npy')
        anomaly=np.load(filename+"anomaly"+str(name)+'.npy')
        return train,test,anomaly
        
    def read_traindata_ocs(self, out_file, Anomaly_file, epoch_num, epoch_size, ratenum):
        #outfile : 読み込むCSVファイル名
        #epoch_num : エポック数
        #epoch_size : エポックサイズ    
        #ratenum : 学習データとテストデータの割合（ここで指定するのはテストデータの割合）
        data = self.read_savedata(out_file)
        anomaly_data = self.read_savedata(Anomaly_file)
        print(data.max(), data.min())
        #前処理正規化
        data = self.preprocessing(data)
        anomaly_data = self.preprocessing(anomaly_data)
        #学習データとテストデータの分割
        rate = 1 - (ratenum/10)
        print("rate", rate)
        RateData = int(data.shape[0] * rate)
        print("data.shape[0]:", data.shape[0])
        print("rate", RateData)
        
        train_data = data[:RateData]
        test = data[RateData:]
        print("TrainData", train_data.shape)
        print("TestData", test.shape)
        print("ÄnomalyDta", anomaly_data.shape)
        #学習用データ、テスト用データ、　異常検知のためのテストデータ
        return train_data, test, anomaly_data