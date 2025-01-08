import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
]


# def normalization(img):
#     img = img / (img.max() - img.min())
#     return img


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Combine3DImages(nn.Module):
    def __init__(self):
        super(Combine3DImages, self).__init__()
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))

    def normalization(self, img):
        img = img / (img.max() - img.min())
        return img

    # def custom_normalize(self, tensor):
    #     # 仅考虑非零元素
    #     non_zero_values = tensor[tensor != 0]
    #
    #     min_val = non_zero_values.min()
    #     max_val = non_zero_values.max()
    #
    #     # 归一化，其中0值仍然为0
    #     normalized_tensor = torch.where(tensor != 0,
    #                                     (tensor - min_val) / (max_val - min_val),
    #                                     tensor)
    #     return normalized_tensor

    def forward(self, image1, image2):
        # 使用自定义归一化
        # image1 = self.normalization(image1)
        # image2 = self.normalization(image2)

        combined = self.weight1 * image1 + image2
        # combined = image1 + image2
        return combined


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        # self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, Y):
        # 计算注意力分数

        # 加权求和得到输出
        output = X * Y

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks, num_classes=2):
        self.inplanes1 = 64
        super(ResNet, self).__init__()
        self.attention = SelfAttention()
        self.combine_images = Combine3DImages()
        self.T1_conv_lv1 = nn.Conv3d(1,
                                     64,
                                     kernel_size=3,
                                     stride=(2, 2, 2),
                                     padding=(1, 1, 1),
                                     bias=False,
                                     dilation=2)
        self.T1_bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.T1_layer1 = self._make_layer(block, 64, blocks=blocks[0], shortcut_type='B', stride=2)
        self.T1_layer2 = self._make_layer(block, 128, blocks=blocks[1], shortcut_type='B', stride=2)
        self.T1_layer3 = self._make_layer(block, 256, blocks=blocks[2], shortcut_type='B', stride=2)
        self.T1_layer4 = self._make_layer(block, 512, blocks=blocks[3], shortcut_type='B', stride=2)
        self.T1_avgpool = nn.AvgPool3d((2, 2, 2), stride=1)

        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes1,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers1 = []
        layers1.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers1.append(block(self.inplanes1, planes))

        return nn.Sequential(*layers1)

    def _make_layer1(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes2,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers2 = []
        layers2.append(block(self.inplanes2, planes, stride, downsample))
        self.inplanes2 = planes * block.expansion
        for i in range(1, blocks):
            layers2.append(block(self.inplanes2, planes))

        return nn.Sequential(*layers2)

    def forward(self, G_2):

        # out = self.combine_images(G_2, W_2)
        out = G_2
        out = self.T1_conv_lv1(out)
        out = self.T1_bn1(out)
        out = self.maxpool(out)
        out = self.T1_layer1(out)
        out = self.T1_layer2(out)
        out = self.T1_layer3(out)
        out = self.T1_layer4(out)
        out = self.T1_avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)

        # out = self.fc1(out)
        # out = F.softmax(out, dim=1)

        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck3d, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck3d, [3, 4, 23, 3], **kwargs)
    return model
