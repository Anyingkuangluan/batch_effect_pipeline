import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import models_vit
from early_stopping import EarlyStopping

import pickle
with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train1.pkl', 'rb') as f:
    train_loader = pickle.load(f)

with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_val/ADNI_val1.pkl', 'rb') as f:
    val_loader = pickle.load(f)


mae = models_vit.vit_base_patch16()

mae = mae.cuda()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(mae.parameters(), lr=0.0002, betas=(0.9, 0.95), weight_decay=0.05)

# load model
mae.load_state_dict(torch.load('./mae3d_gan_wegihts18.pth'), strict=False)

num_epochs = 50

train_losses = []

# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []
save_path = "./" #当前目录下
early_stopping = EarlyStopping(save_path)


for epoch in range(num_epochs):
    epoch_loss = 0.
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data[:, :, 1:, 5:-4, 1:]
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        y = y.to(device)
        # zero grad
        optimizer.zero_grad()
        output = mae(data)
        loss = criterion(output, y.long())

        # backward
        loss.backward()
        # update step
        optimizer.step()
        # print(optimizer.param_groups[0]['lr'])

        iter_loss = loss.item()
        epoch_loss += iter_loss

        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), iter_loss), end='')

    # print(optimizer.param_groups[0]['lr'])
    mean_epoch_loss = epoch_loss / (batch_idx + 1)
    train_losses.append(mean_epoch_loss)
    print("    epoch:",epoch,"loss:",mean_epoch_loss)

    # 早停止
    early_stopping(epoch_loss, mae)
    # 达到早停止条件时，early_stop会被置为True
    if early_stopping.early_stop:
        print("Early stopping")
        break  # 跳出迭代，结束训练

torch.save(mae.state_dict(), './mae3d_class_wegihts18.pth')

print("验证集精度：")
# check prediction
m = len(val_loader.dataset)
batch_size = val_loader.batch_size

y_pred_eegnet = np.zeros(m)
y_true = np.zeros(m)
mae.eval()
with torch.no_grad():
    for batch_idx, (data, y) in tqdm(enumerate(val_loader, 0), total=int(np.ceil(m / batch_size))):
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



print("训练集精度：")
# check prediction
m = len(train_loader.dataset)
batch_size = train_loader.batch_size

y_pred_eegnet = np.zeros(m)
y_true = np.zeros(m)
mae.eval()
with torch.no_grad():
    for batch_idx, (data, y) in tqdm(enumerate(train_loader, 0), total=int(np.ceil(m / batch_size))):
        data = data[:, :, 1:, 5:-4, 1:]
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        # eegnet prediction
        outputs_eegnet = mae(data)
        _, y_pred = torch.max(outputs_eegnet.data, 1)
        y_pred_eegnet[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()

        # # ffn prediction
        # outputs_ffn = ffn(data)
        # _, y_pred = torch.max(outputs_ffn.data, 1)
        # y_pred_ffn[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()

        # labels
        y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()

print("   accuracy {:.5f}%".format((y_true == y_pred_eegnet).sum() / m * 100))


plt.plot(train_losses)
plt.show()
