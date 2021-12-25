import os
import sys
import os.path
import glob
import numpy as np
import datetime as dt
import csv

"指定ディレクトリの全てのファイルを取得（子ディレクトリまで操作）"
#https://www.delftstack.com/ja/howto/python/loop-through-files-in-directory-python/
def file_glob(path, sel = "*"):
    if sel == "*":
        dir = os.path.join(path, '**/*')
        pathlist = glob.glob(dir)
    else:
        dir = os.path.join(path, '**/*.'+sel)
        pathlist = glob.glob(dir)
    return pathlist


#------------csvの読み込み-------------
def get_csvdata(path, numpy_flag = True):
    if numpy_flag:
        list = np.load(path)
    else:
        with open(path) as f:
            reader = csv.reader(f)
            list = [row for row in reader]
            data = np.array([])
            for i,val in enumerate(list):
                try:
                    if i == 1:
                        data = np.array(val)
                        if data.size > 256:
                            data = data[0:256]
                        # print(i, data.shape)
                    else:
                        line = np.array(val)
                        if line.size > 256:
                            line = line[0:256]
                        data = np.vstack((data, line))
                        # print(i, line.shape)
                except ValueError:
                        print("error or end", i, line.shape)
                        pass
    print(data.shape, data.dtype)
    try:
        data = data[:, 0:256]
    except IndexError:
        pass
    return data
#------------csvの読み込み-------------
def get_csv_oneline(path, numpy_flag = False):
    if numpy_flag:
        list = np.load(path)
    else:
        with open(path, "r") as f:
            reader = csv.reader(f)
            list = [row for row in reader]
            list = list[0][0:256]
            list = np.array(list, dtype=np.int64)
            print("list_data")
            print(list.shape, list.dtype)
            print(list)
            # list = f.readlines()[0]
            # list = f.readline()
    list = np.array(list)
    return list

class savedata:
    def __init__(self, path):
        #------入力されるパスは.csvファイルが最後に来るフルパス想定、name = name.csv
        print(path)
        folder, file = os.path.split(path)#フォルダ部分のパスとファイル名を分ける
        self.moviename = file.split('.')[0] #ファイル名の.以前はファイル名である
        name = self.moviename + ".npy"#任意の名前と.csvの拡張子を付与
        self.newpath = os.path.join(folder, name)#新しく作成したファイル名とフォルダを結合
        self.file(folder)#フォルダのパスは存在するか一応確認存在しなかったら作成
    def file(self, path):
        try:
            os.makedirs(path)
            print("make new directory-{}".format(path))
        except FileExistsError:
            pass
#------------------header を入れたいとき-----------------
    def post(self, data):
        np.save(self.newpath, data.astype('float32'))

class ReadCsvTimeSeriesData():
    def __init__(self, folderPath, npyFlag = 0, anomalyData = False):
        """
        folder:CSVの時系列データセットが入っているフォルダを選択OFアプリでは前景の時系列データ＋背景が同じ名前で吐き出されている
        npy:0 全てのでーたで読み込み 1 csv形式でーたで読み込み 2 npy形式でーたで読み込み
        """
        self.flag = npyFlag
        self.anomalyFlag = anomalyData
        self.folderPath = folderPath
        if anomalyData:
            if npyFlag == 1 or npyFlag == 0:
                extName = "csv"
            elif npyFlag == 2:
                extName = "npy"
            self.pathList = file_glob(folderPath)#フォルダ内のpath全て取得
            # path = os.path.join(folderPath, "*."+extName)
            # self.pathList = glob.glob(path)
        else:
            self.pathList = file_glob(folderPath)#フォルダ内のpath全て取得
            if npyFlag != 0:
                if npyFlag == 1:
                    extName = "csv"
                elif npyFlag == 2:
                    extName = "npy"

                pathList2 = []
                for path in self.pathList:
                    ext = path.split('.')[-1]
                    if ext == extName:
                        pathList2.append(path)
                self.pathList = pathList2
                        
            #     path = os.path.join(folderPath, "*.npy")
            #     files = glob.glob(path)
            # else:
            #     path = os.path.join(folderPath, "*.csv")
            #     files = glob.glob(path)
            # self.pathList = files
        if anomalyData:
            self.pathPairList = self.pathList
        else:
            self.pathPairList = []
            for var in self.pathList:
                # name = os.path.splitext(var)#ファイル名と拡張しに分割
                name = var.split('.')
                if name[0][-2:] != "bg":
                    PathPair = [name[0]+"."+name[1], name[0]+"bg"+ "."+name[1]]#輝度分布波形列と背景のペア作成
                    self.pathPairList.append(PathPair)
        # print("DataSetPath:{}".format(self.pathPairList))

    def read_csv(self, newfolder = "", outcsv=False):
        if self.flag == 2:
            pass
        else:
            try:
                if outcsv:
                    pass
                else:
                    outTrain = np.array([])
                    for i in range(len(self.pathPairList)):
                        self.F_y = get_csvdata(self.pathPairList[i][0], numpy_flag=False)
                        self.B_y = get_csv_oneline(self.pathPairList[i][1], numpy_flag=False)
                        for j in range(self.F_y.shape[0]):
                            # self.F_y[i, :] = abs(self.F_y[i, :] - self.B_y)
                            pass
                        print("Shape:{}".format(self.B_y.shape))
                        print("Type:{}".format(type(self.B_y)))
                        print("Data:{}".format(self.B_y))
                        if i == 0:
                            outTrain = self.F_y
                        else:
                            outTrain = np.vstack([outTrain, self.F_y])
                        if newfolder == "":
                            #軽くするためにNPY形式で保存
                            # savemodel = savedata(self.pathPairList[i][0])
                            # savemodel.post(self.F_y)
                            # savemodel = savedata(self.pathPairList[i][1])
                            # savemodel.post(self.B_y)
                            pass
                    print("OK_made_train_data")
                    np.save(self.folderPath + "/train.npy", outTrain.astype('float32'))
                
            except UnicodeDecodeError:
                print("すでにCSVからNPYへの変換は終えています。")
        print("DataSetPath:{}".format(self.pathPairList))
    def out_csv(self, newfolder = ""):
        print("///////////-------------------------------")
        train = []
        for i in range(len(self.pathPairList)):
            diff = []
            self.F_y = get_csvdata(self.pathPairList[i][0], numpy_flag=False)
            self.B_y = get_csv_oneline(self.pathPairList[i][1], numpy_flag=False)
            print("Size:{}".format(self.F_y.shape[0]))
            # diff =  
            
        train = []
        for i in range(len(trainData)):
            data = np.load(str(trainData[i][0]))
            bg = np.load(str(trainData[i][1]))
            # print(data-bg)
            diff = []
            print(data.shape)
            for j in range(data.shape[0]):
                print(data[j,:])
                print(bg)
                diff.append(data[j,:]-bg)
            train.append(diff)
            # print(data[j]-bg)
        train = np.array(train)
        print("DataSetPath:{}".format(self.pathPairList))

    def read_anomaly(self, newfolder = ""):
        self.pathPairList = self.read_difference()
        for i in range(len(self.pathPairList)):
            self.y = get_csv_oneline(self.pathPairList[i], numpy_flag=False)
            if i == 0:
                outTrain = self.y
            else:
                outTrain = np.vstack([outTrain, self.y])
        np.save(self.folderPath + "/anomaly.npy", outTrain.astype('float32'))
        print(self.pathPairList)
        print(outTrain.shape)
        print("OK_made_anomaly_data")
        # savemodel = savedata(self.pathPairList[i])
        # savemodel.post(self.y)

    def read_difference(self, word = "Difference"):
        newlist = []
        for i in self.pathPairList:
            # print(i)
            if i.find("Difference") != -1:
                newlist.append(i)
        # print(newlist)
        return newlist

def readHamamatsuData():
    train = ReadCsvTimeSeriesData("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/DataHamamatsu/1221",npyFlag = 0, anomalyData=False)
    print("///////////////////////////////////////////////")
    train.read_csv()
    anomaly = ReadCsvTimeSeriesData("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/DataHamamatsu/1221anomaly",npyFlag = 0, anomalyData=True)
    # anomalyData.read_anomaly()
    trainData = train.pathPairList
    anomalyData = anomaly.read_difference()
    #TODO:学習データ読み込み形式に合わせて全ての学習データをガッチャンこ
    print("///////////-------------------------------")
    train = []
    for i in range(len(trainData)):
        data = np.load(str(trainData[i][0]))
        bg = np.load(str(trainData[i][1]))
        # print(data-bg)
        diff = []
        print(data.shape)
        for j in range(data.shape[0]):
            print(data[j,:])
            print(bg)
            diff.append(data[j,:]-bg)
        train.append(diff)
        # print(data[j]-bg)
    train = np.array(train)
    anomaly = []
    for i in range(len(anomalyData)):
        diff = np.load(anomalyData[i])
        anomaly.append(diff)
    anomaly = np.array(anomaly)
    print("------------------------------dataSizw-------------------------------")
    print("train:{}, anomaly:{}".format(train.shape, anomaly.shape))

        
    print(trainData)
    print("///////////////////////////////////////////////")
    print(anomalyData)

def readHamamatsuTrain():
    train = ReadCsvTimeSeriesData("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_train",npyFlag = 0, anomalyData=False)
    print("///////////////////////////////////////////////")
    train.read_csv()
    anomaly = ReadCsvTimeSeriesData("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_anomaly",npyFlag = 0, anomalyData=True)
    anomaly.read_anomaly()

readHamamatsuTrain()
# DATA = np.load("/Users/yukihorikawa/Desktop/LAB_LAST/AutoEncoder/AutoEncoder/SensorData/1224NewSensorData/1224Data_train/train.npy")
# print(DATA.shape)
# print(type(DATA))