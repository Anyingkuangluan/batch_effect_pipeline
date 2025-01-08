import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureSimilarityLoss(nn.Module):
    def __init__(self, C3=1e-4, window_size=11, sigma=1.5, epsilon=1e-12):
        """
        初始化结构相似性损失函数。

        Args:
            C3 (float): 稳定分母的小常数。
            window_size (int): 高斯窗口的大小。
            sigma (float): 高斯窗口的标准差。
            epsilon (float): 用于防止数值不稳定的小常数。
        """
        super(StructureSimilarityLoss, self).__init__()
        self.C3 = C3
        self.window_size = window_size
        self.sigma = sigma
        self.epsilon = epsilon

    def gaussian_window(self, window_size, sigma, channels, dims, device):
        """
        创建高斯窗口。

        Args:
            window_size (int): 窗口大小。
            sigma (float): 高斯标准差。
            channels (int): 通道数。
            dims (int): 数据维度（2 或 3）。
            device (torch.device): 设备（CPU 或 GPU）。

        Returns:
            torch.Tensor: 高斯窗口张量。
        """
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        if dims == 2:
            window = g.unsqueeze(1) * g.unsqueeze(0)
            window = window.expand(channels, 1, window_size, window_size).contiguous()
        elif dims == 3:
            window = g.unsqueeze(1) * g.unsqueeze(0) * g.unsqueeze(-1)
            window = window.expand(channels, 1, window_size, window_size, window_size).contiguous()
        return window.to(device)

    def forward(self, x, y):
        """
        前向传播，计算结构相似性损失。

        Args:
            x (torch.Tensor): 预测图像张量，形状为 (N, C, D, H, W) 或 (N, C, H, W)
            y (torch.Tensor): 真实图像张量，形状与 x 相同

        Returns:
            torch.Tensor: 结构相似性损失，标量
        """
        # 确保输入为浮点型
        x = x.type(torch.float32)
        y = y.type(torch.float32)

        # 检查输入维度（支持2D和3D）
        if x.dim() == 5:
            # 3D数据 (N, C, D, H, W)
            conv = F.conv3d
            padding = self.window_size // 2
            dims = 3
        elif x.dim() == 4:
            # 2D数据 (N, C, H, W)
            conv = F.conv2d
            padding = self.window_size // 2
            dims = 2
        else:
            raise ValueError("输入张量必须是4D或5D的 (N, C, H, W) 或 (N, C, D, H, W)")

        N, C = x.size(0), x.size(1)
        device = x.device

        # 创建高斯窗口
        window = self.gaussian_window(self.window_size, self.sigma, C, dims, device)

        # 计算均值
        mu_x = conv(x, window, padding=padding, groups=C)
        mu_y = conv(y, window, padding=padding, groups=C)

        # 计算协方差和方差
        sigma_xy = conv(x * y, window, padding=padding, groups=C) - mu_x * mu_y
        sigma_x = conv(x * x, window, padding=padding, groups=C) - mu_x * mu_x
        sigma_y = conv(y * y, window, padding=padding, groups=C) - mu_y * mu_y

        # Clamp 方差，确保非负且有下限
        sigma_x = torch.clamp(sigma_x, min=self.epsilon)
        sigma_y = torch.clamp(sigma_y, min=self.epsilon)

        # 计算结构相似性
        structure = (sigma_xy + self.C3) / (torch.sqrt(sigma_x) * torch.sqrt(sigma_y) + self.C3)

        # 取平均结构相似性作为损失
        structure_mean = structure.view(N, C, -1).mean(dim=2)

        # 计算损失（1 - SSIM_structure）
        loss = 1 - structure_mean
        loss = loss.mean()  # 对所有样本和通道取平均

        return loss
