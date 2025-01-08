import numpy as np
from torch import nn
import torch
import datetime
import lijinjin.gen_model as gen_models
import random
from SSIM_3 import StructureSimilarityLoss
from module import NEW_data_set
from module import pkl_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# data_set

root = "D:/data/"
label1 = "ADNI"
label2 = "HCP"
label3 = "OASIS"
label4 = "NACC"
dir1 = "data_10_fold/ADNI_output.csv"
dir2 = "data_10_fold/HCP_output.csv"
dir3 = "data_10_fold/OASIS_output.csv"
dir4 = "data_10_fold/NACC_output.csv"

pkl_1_1 = "C:/Users/Administrator/PycharmProjects/ADNI_merged_data_ALL.pkl"

pkl_2_1 = "C:/Users/Administrator/PycharmProjects/HCP_merged_data_ALL.pkl"

pkl_3_1 = "C:/Users/Administrator/PycharmProjects/OASIS_merged_data_ALL.pkl"

pkl_4_1 = "C:/Users/Administrator/PycharmProjects/NACC_merged_data_ALL.pkl"

batch_size = 1
dataset1 = NEW_data_set.MyDataset(dir1, label1, root)
print(len(dataset1))
dataset2 = NEW_data_set.MyDataset(dir2, label2, root)
print(len(dataset2))
dataset3 = NEW_data_set.MyDataset(dir3, label3, root)
print(len(dataset3))
dataset4 = NEW_data_set.MyDataset(dir4, label4, root)
print(len(dataset4))
all_dataset_1 = dataset1 + dataset2 + dataset3 + dataset4
# all_dataset_1 = dataset1

dataset_1 = pkl_dataset.CustomDataset(pkl_1_1)

print(len(dataset_1))
dataset_2 = pkl_dataset.CustomDataset(pkl_2_1)

print(len(dataset_2))
dataset_3 = pkl_dataset.CustomDataset(pkl_3_1)

print(len(dataset_3))
dataset_4 = pkl_dataset.CustomDataset(pkl_4_1)

print(len(dataset_4))
target_all_dataset_1 = dataset_1 + dataset_2 + dataset_3 + dataset_4
# target_all_dataset_1 = dataset_1

# dataset = dataset1
now = datetime.datetime.now()

mae = gen_models.mae_vit_base_patch16_dec512d8b()
mae = mae.cuda()

batch_size = 1
run = 1
optimizer_g = torch.optim.AdamW(mae.parameters(), lr=0.0002, weight_decay=0.05)
# optimizer_g_1 = torch.optim.SGD()
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
criterion = nn.MSELoss().to(device)
ssim_metric = StructureSimilarityLoss().to(device)
eval_losses = []
eval_acces = []
# early_stopping = EarlyStopping(save_path)
padding = [8, 8, 0, 0, 8, 8]
num_epochs = 80
for fold in range(13, 14):
    now = datetime.datetime.now()
    print(now)
    seed = fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.10, random_state=fold)
    train_loader = torch.utils.data.DataLoader(all_dataset_1, batch_size=batch_size, shuffle=False)
    target_loader = torch.utils.data.DataLoader(target_all_dataset_1, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    minloss = 100

    for epoch in range(num_epochs):
        now = datetime.datetime.now()
        epoch_loss = 0.
        num_batch = 0
        for (data1, labels1), (data2, labels2) in zip(train_loader, target_loader):
            random_integer = random.randint(0, 3)
            imgs1 = data1
            # print(imgs1.shape)
            target1 = labels1
            target2 = labels2
            rebuild_target = data1
            # print(rebuild_target)
            # print(target1)
            real_data = imgs1
            # print(real_data.shape)
            real_data = real_data[:, :, 1:, 5:-4, 1:]
            # real_data = F.pad(real_data, pad=padding, mode='constant', value=0)  # (113,137,113) F.pad(x, pad=padding, mode='constant', value=0)
            # print(real_data.shape)
            rebuild_target = rebuild_target[:, :, 1:, 5:-4, 1:]
            # rebuild_target = F.pad(rebuild_target, pad=padding, mode='constant', value=0)
            # print(rebuild_target.shape)
            if random_integer == 0:
                # 0°
                rebuild_target = rebuild_target
                real_data = real_data
            if random_integer == 1:
                # z90°
                rebuild_target = torch.rot90(rebuild_target, k=2, dims=(3, 2))
                real_data = torch.rot90(real_data, k=2, dims=(3, 2))
            if random_integer == 2:
                # x90°
                rebuild_target = torch.rot90(rebuild_target, k=2, dims=(4, 3))
                real_data = torch.rot90(real_data, k=2, dims=(4, 3))
            if random_integer == 3:
                # y90°
                rebuild_target = torch.rot90(rebuild_target, k=2, dims=(4, 2))
                real_data = torch.rot90(real_data, k=2, dims=(4, 2))
            # print(real_data.shape)
            rebuild_target = rebuild_target.float()
            rebuild_target = rebuild_target.to(device)
            real_data = real_data.float()
            real_data = real_data.to(device)
            # target1 = target1.float()
            # target1 = target1
            # target2 = target2.numpy()
            target1 = target1.cuda()
            # print(target1)
            # target2 = target2.cuda()
            optimizer_g.zero_grad()
            # x1 = mae(real_data)
            # print(x1)
            rebuild_data, x1, x2 = mae(real_data)
            # print(rebuild_data)
            # print(rebuild_data.sha
            loss1 = criterion(rebuild_data, rebuild_target)
            loss2 = 1-ssim_metric(rebuild_data, rebuild_target)
            # print(loss1, loss2)
            x2 = x2.unsqueeze(0)
            # print(loss1, pred, mask, label1, label2)
            loss3 = loss_fn(x2, target2)
            loss4 = loss_fn(x1, target1)

            # loss3 = loss_fn(x2, target2)

            loss = loss1 + loss2 + loss3 + loss4

            loss.backward()
            optimizer_g.step()

            iter_loss = loss.item()

            epoch_loss += iter_loss

            num_batch = num_batch + 1
        mean_epoch_loss = epoch_loss / (num_batch + 1)

        torch.save(mae.state_dict(), 'weights/White_mae3d_gan_wegihts_{}_{}.pth'.format(fold, epoch))
        #
        # model = mae
        # model.load_state_dict(torch.load('weights/White_mae3d_gan_wegihts_{}.pth'.format(fold)))
        # model.eval()
        # with torch.no_grad():
        #     for data in test_loader:
        #         imgs2, target2 = data
        #
        #         real_data_1 = imgs2
        #         # print(real_data.shape)
        #         real_data_1 = real_data_1[:, :, 1:, 5:-4, 1:]  # (112,128,112)
        #         real_data_1 = real_data_1.float()
        #         real_data_1 = real_data_1.to(device)
        #         target2 = target2.cuda()
        #         # 更新生成器
        #         # x1 = mae(real_data_1)
        #         loss1, x2 = mae(real_data_1)
        #         # x1 = x1
        #         x2 = x2.unsqueeze(0)
        #         loss2 = loss_fn(x2, target2)
        #
        #         loss = loss1 + loss2
        #
        #         iter_loss = loss.item()
        #         epoch_loss += iter_loss
        #
        #         num_batch = num_batch + 1
        #     mean_epoch_loss = epoch_loss / (num_batch + 1)
        #     print(" epoch:", epoch, "测试生成器loss:", mean_epoch_loss)
