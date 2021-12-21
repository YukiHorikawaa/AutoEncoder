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
    data = data[:, 0:256]
    return data
#------------csvの読み込み-------------
def get_csv_oneline(path, numpy_flag = True):
    if numpy_flag:
        list = np.load(path)
    else:
        with open(path, "r") as f:
            reader = csv.reader(f)
            list = [row for row in reader]
            # list = f.readlines()[0]
            # list = f.readline()
            list = list[0][0:256]
    list = np.array(list)
    print("list_data")
    # print(list)
    print(list.shape, list.dtype)
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
        np.save(self.newpath, data)

class ReadCsvTimeSeriesData():
    def __init__(self, folderPath, npyFlag = 0):
        """
        folder:CSVの時系列データセットが入っているフォルダを選択OFアプリでは前景の時系列データ＋背景が同じ名前で吐き出されている
        npy:0 全てのでーたで読み込み 1 csv形式でーたで読み込み 2 npy形式でーたで読み込み
        """
        self.flag = npyFlag
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
    
        self.pathPairList = []
        for var in self.pathList:
            # name = os.path.splitext(var)#ファイル名と拡張しに分割
            name = var.split('.')
            if name[0][-2:] != "bg":
                PathPair = [name[0]+"."+name[1], name[0]+"bg"+ "."+name[1]]#輝度分布波形列と背景のペア作成
                self.pathPairList.append(PathPair)
        print("DataSetPath:{}".format(self.pathPairList))

    def read_csv(self, newfolder = ""):
        if self.flag == 2:
            pass
        else:
            try:
                for i in range(len(self.pathPairList)):
                    self.F_y = get_csvdata(self.pathPairList[i][0], numpy_flag=False)
                    self.B_y = get_csv_oneline(self.pathPairList[i][1], numpy_flag=False)
                    print("Size:{}".format(self.F_y.shape[0]))
                    if newfolder == "":
                        #軽くするためにNPY形式で保存
                        savemodel = savedata(self.pathPairList[i][0])
                        savemodel.post(self.F_y)
                        savemodel = savedata(self.pathPairList[i][1])
                        savemodel.post(self.B_y)
            
            except UnicodeDecodeError:
                print("すでにCSVからNPYへの変換は終えています。")
        print("DataSetPath:{}".format(self.pathPairList))

model = ReadCsvTimeSeriesData("/Users/yukihorikawa/Downloads/test",npyFlag = 1)
model.read_csv()
