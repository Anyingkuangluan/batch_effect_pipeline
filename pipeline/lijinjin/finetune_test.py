import torch
import numpy as np
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import models_vit

import pickle

with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test1.pkl', 'rb') as f:
    test_loader = pickle.load(f)

mae = models_vit.vit_base_patch16()

mae = mae.cuda()

# load model
mae.load_state_dict(torch.load('./mae3d_class_wegihts18.pth'), strict=False)

print("测试集精度：")
# check prediction
m = len(test_loader.dataset)
batch_size = test_loader.batch_size

y_pred_eegnet = np.zeros(m)
y_true = np.zeros(m)
with torch.no_grad():
    for batch_idx, (data, y) in tqdm(enumerate(test_loader, 0), total=int(np.ceil(m / batch_size))):
        data = data[:, :, 1:, 5:-4, 1:]
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        # eegnet prediction
        outputs_eegnet = mae(data)

        _, y_pred = torch.max(outputs_eegnet.data, 1)
        # print("batch_size",batch_size)


        y_pred_eegnet[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()


        # labels
        y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()


print("   accuracy {:.5f}%".format((y_true == y_pred_eegnet).sum() / m * 100))
print(y_true)
print(y_pred_eegnet)