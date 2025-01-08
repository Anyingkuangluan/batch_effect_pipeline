import numpy as np
import pandas as pd
import torch.utils.data
import random
import time
import module.OUR_pkl_dataset as dataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast
import our_model as our_model
import module.NEW_data_set as NEW_data_set
from sklearn.model_selection import train_test_split

root = "D:/多中心大规模数据/ADNI_GM_1/"
NC_label = "NC"
ad_label = "AD"
nc_train_data_dir = "data_10_fold/cn_data.csv"
ad_train_data_dir = "data_10_fold/ad_data.csv"
NC_pkl_file1 = "data_10_fold/CN_data.pkl"
AD_pkl_file1 = "data_10_fold/AD_data.pkl"
cn_dataset_1 = dataset.CustomDataset(NC_pkl_file1)
ad_dataset_1 = dataset.CustomDataset(AD_pkl_file1)
dataset_1 = cn_dataset_1 + ad_dataset_1

batch_size = 1
cn_dataset = NEW_data_set.MyDataset(nc_train_data_dir, NC_label, root)
ad_dataset = NEW_data_set.MyDataset(ad_train_data_dir, ad_label, root)
dataset = cn_dataset + ad_dataset

acc = []
val_max_acc = []

# 0 1
for fold in range(10):
    seed = fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dataset_1, test_dataset_1 = train_test_split(dataset, test_size=0.10, random_state=seed)
    train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size)
    test_loader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False)

    train_dataset_2, test_dataset_2 = train_test_split(dataset_1, test_size=0.10, random_state=seed)
    train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size)
    test_loader_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # 设置学习参数
    num_classes = 2
    learning_rate = 1e-3
    epoch = 30
    # 设置损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 设置模型
    model = our_model.resnet34(num_classes=num_classes).to(device)
    # 设置激活函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # 设置统计数据初始值
    max_auccracy = 0
    min_test_loss = 10000
    min_train_loss = 20000
    train_step = 0
    model.train()
    # 开始训练
    loss_change = []

    for i in range(epoch):
        # 输出训练时间点
        print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
        total_loss1 = 0
        start_time = time.time()
        for (data1, target1), (gray_data, white_data) in zip(train_loader_1, train_loader_2):
            imgs1 = data1
            target1 = target1.to(device)

            imgs1 = imgs1.permute(1, 0, 2, 3, 4, 5).float()
            gray_origin = imgs1[0]
            White_origin = imgs1[1]
            gray_data_1 = gray_origin[:, :, 1:, 5:-4, 1:]
            White_data_1 = White_origin[:, :, 1:, 5:-4, 1:]

            gray_data_1_1 = gray_data_1.to(device).float()
            White_data_1_1 = White_data_1.to(device).float()
            data1_1 = 0.7705 * gray_data_1_1 + White_data_1_1

            gray_data_2_1 = gray_data.unsqueeze(0).to(device).float()
            white_data_2_1 = white_data.unsqueeze(0).to(device).float()
            data2_1 = 0.7705 * gray_data_2_1 + white_data_2_1
            # print(data1_1.shape, data2_1.shape)
            outputs1 = model(data1_1, data2_1)
            # print(outputs1)
            loss1 = loss_fn(outputs1, target1)

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            train_step = train_step + 1
            total_loss1 = total_loss1 + loss1.item()
            if train_step % 100 == 0:
                print(f'Epoch {i + 1}, Loss: {loss1.item()}')
            if total_loss1 <= min_train_loss:
                min_train_loss = total_loss1
                torch.save(model.state_dict(),
                           "result_model/新模型病理分类_finaltest_minloss_model_{}.pth".format(fold + 1))
        loss_change.append(total_loss1)
        scheduler.step()
        print("CN_vs_AD_GM_NEW_train_loss: {}  {} times".format(total_loss1, fold + 1))
        print("CN_vs_AD_GM_NEW_min_train_loss:{}  {} times".format(min_train_loss, fold + 1))

    csv_name_loss = ["loss"]
    acc_list = pd.DataFrame(data=loss_change, columns=csv_name_loss)
    acc_list.to_csv('result_csv/新模型病理分类_finaltest_loss_{}.csv'.format(fold + 1), encoding='gbk',
                    index=False)

    # 10折验证测试
    model = our_model.resnet34(num_classes=num_classes).to(device)
    model.load_state_dict(
        torch.load('result_model/新模型病理分类_finaltest_minloss_model_{}.pth'.format(fold + 1)))
    model = model.to(device)

    label_list = []
    pred_list = []
    model.eval()
    auccracy = 0
    with torch.no_grad():
        for (data1, target1), (gray_data, white_data) in zip(test_loader_1, test_loader_2):
            # 单独模型测试
            imgs3 = data1
            target3 = target1
            imgs3 = imgs3.permute(1, 0, 2, 3, 4, 5).float()
            gray_origin_3 = imgs3[0]
            White_origin_3 = imgs3[1]
            gray_data_3 = gray_origin_3[:, :, 1:, 5:-4, 1:]
            White_data_3 = White_origin_3[:, :, 1:, 5:-4, 1:]

            gray_data_3_1 = gray_data_3.to(device).float()
            White_data_3_1 = White_data_3.to(device).float()
            data1_2 = 0.7705 * gray_data_3_1 + White_data_3_1

            gray_data_3_2 = gray_data.unsqueeze(0).to(device).float()
            white_data_3_2 = white_data.unsqueeze(0).to(device).float()
            data2_2 = 0.7705 * gray_data_3_2 + white_data_3_2

            target3 = target3.to(device)
            outputs3 = model(data1_2, data2_2)

            # 数据保存
            pred_softmax = torch.softmax(outputs3, 1).cpu().numpy()
            outputs3 = outputs3.argmax(1)
            if outputs3 == target3:
                auccracy = auccracy + 1

            pred_list.append(pred_softmax.tolist()[0])
            label_true = ["label"]
            df_label = pd.DataFrame(data=label_list, columns=label_true)
            df_label.to_csv('result_csv/新模型病理分类_finaltest_label_{}.csv'.format(fold + 1), encoding='gbk',
                            index=False)
            label_name = ["CN", "AD"]
            df_pred = pd.DataFrame(data=pred_list, columns=label_name)
            df_pred.to_csv('result_csv/新模型病理分类_finaltest_pred_result_{}.csv'.format(fold + 1),
                           encoding='gbk', index=False)
        auc_rat = auccracy / len(test_dataset_1) * 100
        print("CN_vs_AD_GM_NEW_Auccracy in test data: {}%  {} times".format(auc_rat, fold + 1))

        acc.append(auc_rat)

# 保存模型每折准确率
csv_name = ["acc"]
acc_list = pd.DataFrame(data=acc, columns=csv_name)
acc_list.to_csv('result_csv/新模型病理分类_finall_acc_all.csv', encoding='gbk', index=False)
# 保存每次验证的准确率
csv_name_val_acc = ['acc']
acc_list = pd.DataFrame(data=val_max_acc, columns=csv_name_val_acc)
acc_list.to_csv('result_csv/新模型病理分类_val_acc_all.csv', encoding='gbk', index=False)
