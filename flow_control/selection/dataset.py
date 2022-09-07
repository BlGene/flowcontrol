import numpy as np

data1 = np.load('./selection/dataset_100_105.npz')['arr_0']
data2 = np.load('./selection/dataset_105_110.npz')['arr_0']
data3 = np.load('./selection/dataset_110_115.npz')['arr_0']
data4 = np.load('./selection/dataset_115_120.npz')['arr_0']

data = np.vstack((data1, data2, data3, data4))
print(data.shape, np.max(data), np.min(data), np.sum(data))

np.savez('full_dataset_test.npz', data)