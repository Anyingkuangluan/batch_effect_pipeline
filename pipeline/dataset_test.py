import numpy as np
import pandas as pd
import torch.utils.data
import random
import pickle
import gc
import time
import model.dataset as dataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import lijinjin.rebuild_image as gen_models
import numpy as np
import matplotlib.pyplot as plt
import model.resnet3D as GM_NEW_resnet3D
import module.NEW_data_set as NEW_data_set
from sklearn.model_selection import train_test_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Gray_pretrain_model = gen_models.mae_vit_base_patch16_dec512d8b()
Gray_pretrain_model.load_state_dict(torch.load('weights/Gray_mae3d_gan_wegihts.pth'))

# data_set制作
for i in range(1):
    pkl_files = "D:/processed_data/NEW_data_set/Gray_processed_data_{}.pkl".format(i + 1)

    # 创建自定义数据集实例
    print('开始制作data{}'.format(i+1))
    with open(pkl_files, 'rb') as file:
        data = pickle.load(file)
    # print('开始制作data2')
    # dataset2 = dataset.CustomDataset(pkl_files[1])
    # print('开始制作data3')
    # dataset3 = dataset.CustomDataset(pkl_files[2])
    # print('开始制作data4')
    # dataset4 = dataset.CustomDataset(pkl_files[3])

    # dataset = dataset4


    big_data_processed_data = []
    i = 1
    for data1 in data:
        imgs1, target1, target2 = data1
        imgs1 = torch.from_numpy(imgs1)
        imgs1 = imgs1.unsqueeze(0)
        gray_data_1 = imgs1[:, :, 1:, 5:-4, 1:]
        gray_data_1_1 = gray_data_1.float()
        gray_data_1_1 = gray_data_1_1

        gray_data_1_2 = Gray_pretrain_model(gray_data_1_1)

        gray_data_1_2 = gray_data_1_2.cpu()
        gray_data_1_2 = gray_data_1_2.squeeze(0).squeeze(0)
        gray_data_1_2 = gray_data_1_2.detach().numpy()
        # print(gray_data_1_2.shape)
        direct = (1, 2, 0)  # (2, 1, 0) axis   (1, 2, 0) coronal    (0, 2, 1) sagittal
        gray_data_1_2 = np.transpose(gray_data_1_2, direct)
        gray_data_1_2 = np.flip(gray_data_1_2, axis=0)
        slice_2d = gray_data_1_2[:, 60, :]
        plt.imshow(slice_2d, cmap='jet', interpolation='nearest')
        plt.colorbar()  # 显示颜色条
        plt.title('Heatmap of 2D Slice from 3D Data')
        plt.show()
    # with open("D:/processed_data/NEW_data_set/new/Gray_attention_processed_data_{}.pkl".format(i + 1), 'wb') as f:
    #     pickle.dump(big_data_processed_data, f)
    #
    # gc.collect()
