
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import numpy as np

import rebuild_image
# load data
import pickle
with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test1.pkl', 'rb') as f:
    test_loader = pickle.load(f)
# NACC HCP OASIS
# with open('/home/manjianzhi/jinjin/HCP_pkl/HCP_test_100.pkl', 'rb') as f:
#     test_loader = pickle.load(f)


# 初始化resnetgan网络，用于特征提取
mae = rebuild_image.mae_vit_base_patch16_dec512d8b()
mae = mae.cuda()

# load model
mae.load_state_dict(torch.load('./mae3d_gan_wegihts18.pth'))
mae.eval()
real_datas = []
fake_datas = []
label = []
with torch.no_grad():
    for batch_idx, (real_data, y) in enumerate(test_loader):
        real_data = real_data[:, :, 1:, 5:-4, 1:]
        real_data = real_data.type(torch.FloatTensor)
        real_data = real_data.to(device)
        fake_data = mae(real_data)

        real_data = real_data[0][0]
        fake_data = fake_data[0][0]

        real_datas.append(real_data.cpu().numpy())
        fake_datas.append(fake_data.cpu().numpy())

        y = y[0].numpy()
        label.append(y)


# 保存数组到文件（使用np.save函数）
np.save('/home/manjianzhi/jinjin/MAE/real_fake_data3d_3/ADNI_test_real_datas_resnet.npy', np.array(real_datas))
np.save('/home/manjianzhi/jinjin/MAE/real_fake_data3d_3/ADNI_test_fake_datas_resnet.npy', np.array(fake_datas))
np.save('/home/manjianzhi/jinjin/MAE/real_fake_data3d_3/ADNI_test_label_resnet.npy', np.array(label))
