import os.path
import random
import nibabel as nib
import numpy as np
import pandas as pd
import torch.utils.data
from scipy import stats
from torch.utils.data import Dataset



seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def z_score_normalize(matrix):
    """
    对三维矩阵进行Z-Score归一化。
    :param matrix: 一个三维NumPy数组。
    :return: Z-Score归一化后的矩阵。
    """
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean) / std

# 输入归一化
# def normalization(img):
#     img = img / (img.max() - img.min())
#     return img

#
class MyDataset(Dataset):
    def __init__(self, csv, label, root):
        self.Gray_data = []
        self.White_data = []
        self.ALL_data = []
        df = pd.read_csv(csv)
        self.Gray_data = df['G'].tolist()
        self.White_data = df['W'].tolist()
        self.ALL_data = df['ALL'].tolist()
        self.label = label
        self.root = os.path.join(root, label)

    def __getitem__(self, index):  # 检索函数
        # 数据获取
        Gray_data_name, White_data_name = self.Gray_data[index], self.White_data[index]
        Gray_data_path = os.path.join(self.root, Gray_data_name)
        White_data_path = os.path.join(self.root, White_data_name)
        Gray_imgs = nib.load(Gray_data_path)
        White_imgs = nib.load(White_data_path)
        Gray_imgs = Gray_imgs.get_fdata()
        White_imgs = White_imgs.get_fdata()

        # 数据小数点后处理
        Gray_imgs = np.around(Gray_imgs, 4)
        Gray_imgs[Gray_imgs < 0.2] = 0
        White_imgs = np.around(White_imgs, 4)
        White_imgs[White_imgs < 0.2] = 0
        # print('index')
        # Gray_imgs = normalization(Gray_imgs)
        # White_imgs = normalization(White_imgs)

        # 数据转换及输出
        Gray_imgs = torch.from_numpy(Gray_imgs)
        White_imgs = torch.from_numpy(White_imgs)
        Gray_imgs = torch.reshape(Gray_imgs, (1, 113, 137, 113))
        White_imgs = torch.reshape(White_imgs, (1, 113, 137, 113))
        G_W_img_data = torch.stack((Gray_imgs, White_imgs))

        # ALL data
        ALL_data_name = self.ALL_data[index]
        ALL_data_path = os.path.join(self.root, ALL_data_name)
        ALL_imgs = nib.load(ALL_data_path)
        ALL_imgs = ALL_imgs.get_fdata()

        # 数据小数点后处理
        ALL_imgs = np.around(ALL_imgs, 4)
        ALL_imgs[ALL_imgs < 0.2] = 0

        # 数据转换及输出
        ALL_imgs = torch.from_numpy(ALL_imgs)
        ALL_imgs = torch.reshape(ALL_imgs, (1, 1, 113, 137, 113))
        ALL_img_data = ALL_imgs

        label = 0
        if self.label == "AD":
            label = 0
        if self.label == "NC":
            label = 1
        return ALL_img_data, label

    def __len__(self):
        return len(self.Gray_data)
