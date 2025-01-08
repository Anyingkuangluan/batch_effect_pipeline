from torch.utils.data import Dataset, DataLoader
import pickle
import torch


class CustomDataset(Dataset):
    def __init__(self, pkl_file):
        # 加载数据
        if not isinstance(pkl_file, str):
            raise TypeError(f"Invalid path type: {type(pkl_file)}. Expected str.")
        with open(pkl_file, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        # 返回数据集中的数据项数量
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引idx获取数据
        # 假设每个数据项是 (图像, label1, label2)
        image1, label1, label2 = self.data[idx]


        return (image1, label1, label2)