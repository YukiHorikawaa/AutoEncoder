import os
import glob #特定のディレクトリに存在するファイルに処理を加える
import csv
import pandas as pd
class concat():
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

    def output_data(self, outFile_name):
        list=[]
        for path in self.All_Files:
            data_df = pd.read_csv(path, header=None, engine="python")
            data = data_df.iloc[3:, 3].reset_index(drop = True).values
            list.append(data)
        frame=pd.DataFrame(list)
        out = outFile_name+".csv"
        self.test_data_path = os.path.join(self.Folder_PATH, "test_data", out)
        frame.to_csv(self.test_data_path, index=False, encoding="utf-8")  

        data_df = pd.read_csv(self.test_data_path, engine="python")
        data = data_df.values
        print(data.shape)
        print(data[0])

    def get_data(self, out_file):
        out = out_file+".csv"
        data_path = os.path.join(self.Folder_PATH, "test_data", out)
        data_df = pd.read_csv(data_path, engine="python")
        return data_df