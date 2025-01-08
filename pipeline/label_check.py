import torch
from torch.utils.data import DataLoader

import model.dataset as dataset


pkl_files = ['D:/processed_data/Gray_processed_data_0.pkl', 'D:/processed_data/Gray_processed_data_1.pkl',
             'D:/processed_data/Gray_processed_data_2.pkl', 'D:/processed_data/Gray_processed_data_3.pkl']

# 创建自定义数据集实例
print('开始制作data1')
dataset1 = dataset.CustomDataset(pkl_files[0])
print('开始制作data2')
dataset2 = dataset.CustomDataset(pkl_files[1])
print('开始制作data3')
dataset3 = dataset.CustomDataset(pkl_files[2])
print('开始制作data4')
dataset4 = dataset.CustomDataset(pkl_files[3])

dataset = dataset1 + dataset2 + dataset3 + dataset4

loader = DataLoader(dataset, batch_size=1, shuffle=True)
# 收集所有标签
all_labels1 = []
all_labels2 = []
for _, labels1, labels2 in loader:
    all_labels1.extend(labels1)
    all_labels2.extend(labels2)
print(all_labels1, all_labels2)
