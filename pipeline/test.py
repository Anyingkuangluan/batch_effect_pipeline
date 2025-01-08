import torch
import torch.nn as nn

# 定义数据和目标
x1 = torch.tensor([[-0.0850, 1.9281, -0.2269, 0.2636]], requires_grad=True)
target1 = torch.tensor([0], dtype=torch.long)

# 定义损失函数并计算损失
loss_fn = nn.CrossEntropyLoss()
loss2 = loss_fn(x1, target1)
print("Loss:", loss2.item())