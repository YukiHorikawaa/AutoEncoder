import dataset
Dataset = dataset.dataset("Obrid_AE", "data")
Dataset.concat_data("sample_data")
Dataset = dataset.dataset("Obrid_AE", "test")
print("----------------------")
Dataset.concat_data("sample_test")

print("----------------------")
data = Dataset.read_savedata("sample_test")
print(data.shape[0])
print("----------------------")
data, test_data , anomaly_data= Dataset.read_traindata("sample_data", "sample_test", 50, 200, 2)
