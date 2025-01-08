import numpy as np
import pandas as pd
import torch.utils.data
import random
import time

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import lijinjin.rebuild_image as gen_models

import model.resnet3D as GM_NEW_resnet3D
import module.NEW_data_set as NEW_data_set
from sklearn.model_selection import train_test_split
import module.NEW_out_data_set as NEW_out_data_set

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


root = "D:/多中心大规模数据/AIBL/"
NC_label = "out_NC"
AD_label = "out_AD"
nc_train_data_dir = "data_10_fold/AIBL_NC_data_145_1.csv"
AD_train_data_dir = "data_10_fold/AIBL_AD_data_1.csv"
batch_size = 1
cn_dataset = NEW_out_data_set.MyDataset(nc_train_data_dir, NC_label, root)
AD_dataset = NEW_out_data_set.MyDataset(AD_train_data_dir, AD_label, root)
out_test_dataset = cn_dataset + AD_dataset
test_loader = torch.utils.data.DataLoader(out_test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc = []
for fold in range(0,30):
    model = GM_NEW_resnet3D.resnet10()
    model.load_state_dict(
        torch.load('result_model/CN_vs_AD_GM_NEW_ResNet10_finaltest_minloss_model_{}.pth'.format(fold + 1)))
    model = model.to(device)

    Gray_pretrain_model = gen_models.mae_vit_base_patch16_dec512d8b().to(device)
    White_pretrain_model = gen_models.mae_vit_base_patch16_dec512d8b().to(device)
    Gray_pretrain_model.load_state_dict(torch.load('weights/Gray_mae3d_gan_wegihts.pth'))
    White_pretrain_model.load_state_dict(torch.load('weights/White_mae3d_gan_wegihts.pth'))

    label_list = []
    pred_list = []
    model.eval()
    auccracy = 0
    with torch.no_grad():
        for data in test_loader:
            # 单独模型测试
            imgs3, target3 = data
            imgs3 = imgs3.permute(1, 0, 2, 3, 4, 5).float()
            gray_origin_3 = imgs3[0]
            White_origin_3 = imgs3[1]
            gray_data_3 = gray_origin_3[:, :, 1:, 5:-4, 1:]
            White_data_3 = White_origin_3[:, :, 1:, 5:-4, 1:]
            # gray_data = gray_data.float()
            # White_data = White_data.float()
            gray_data_3_1 = gray_data_3.to(device).float()
            White_data_3_1 = White_data_3.to(device).float()

            gray_data_3_2 = Gray_pretrain_model(gray_data_3_1)
            White_data_3_2 = White_pretrain_model(White_data_3_1)
            target3 = target3.to(device)
            outputs3 = model(gray_data_3_1, White_data_3_1, gray_data_3_2, White_data_3_2)

            # 数据保存
            pred_softmax = torch.softmax(outputs3, 1).cpu().numpy()
            outputs3 = outputs3.argmax(1)
            if outputs3 == target3:
                auccracy = auccracy + 1
            # else:
            #     print(outputs3)
            pred_list.append(pred_softmax.tolist()[0])
            label_true = ["label"]
            df_label = pd.DataFrame(data=label_list, columns=label_true)
            df_label.to_csv('result_csv/CN_vs_AD_GM_NEW_ResNet10_AIBL_finaltest_label_{}.csv'.format(fold + 1), encoding='gbk', index=False)
            label_name = ["CN", "AD"]
            df_pred = pd.DataFrame(data=pred_list, columns=label_name)
            df_pred.to_csv('result_csv/CN_vs_AD_GM_NEW_ResNet10_AIBL_finalpred_result_{}.csv'.format(fold + 1), encoding='gbk', index=False)
        auc_rat = auccracy / len(out_test_dataset) * 100
        print("CN_vs_AD_GM_NEW_Auccracy in AIBL test data: {}%  {} times".format(auc_rat, fold + 1))

        acc.append(auc_rat)

# 保存模型每折准确率
csv_name = ["acc"]
acc_list = pd.DataFrame(data=acc, columns=csv_name)
acc_list.to_csv('result_csv/CN_vs_AD_GM_NEW_ResNet10_AIBL_finalacc_all_2.csv', encoding='gbk', index=False)
