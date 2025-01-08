import torch
from torch import nn


class EnhancedLeakyReLU(nn.Module):
    def __init__(self, alpha_positive=0.01, alpha_negative=1):
        """
        Enhanced Leaky ReLU Activation.

        Parameters:
        - alpha_positive: 斜率，用于正值部分，一般小于1。
        - alpha_negative: 斜率，用于负值部分，一般大于1。
        """
        super(EnhancedLeakyReLU, self).__init__()
        self.alpha_positive = alpha_positive
        self.alpha_negative = alpha_negative

    def forward(self, x):
        """
        Forward pass of the activation function.

        Parameters:
        - x: 输入张量。

        Returns:
        - 激活后的张量。
        """
        # 对正值应用alpha_positive系数
        positive_part = torch.where(x > 0, x * self.alpha_positive, x * 0)
        # 对负值应用alpha_negative系数
        negative_part = torch.where(x < 0, x * self.alpha_negative, x * 0)
        return positive_part + negative_part