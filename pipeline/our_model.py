import time
import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super(BasicBlock3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
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

class InterAttention(nn.Module):
    def __init__(self, in_dim, heads=1, dropout_rate=0.1):
        super(InterAttention, self).__init__()
        self.attention = nn.MultiheadAttention(in_dim, heads, dropout=dropout_rate)

    def forward(self, query, key):
        if query.dim() != 5:
            raise ValueError(
                f"Expected input with 5 dimensions (batch, channels, depth, height, width), but got {query.dim()} dimensions"
            )

        batch, channels, depth, height, width = query.size()

        query = query.view(batch, channels, -1).permute(2, 0, 1)  # (L, N, E)
        key = key.view(batch, channels, -1).permute(2, 0, 1)      # (S, N, E)
        value = key

        x, _ = self.attention(query, key, value)  # x: (L, N, E)

        x = x.permute(1, 2, 0).view(batch, channels, depth, height, width)
        return x

class ResNetStream(nn.Module):
    def __init__(self, block, layers):
        super(ResNetStream, self).__init__()

        self.inplanes_x1 = 64
        self.inplanes_x2 = 64

        self.conv1_x1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_x1 = nn.BatchNorm3d(64)

        self.conv1_x2 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_x2 = nn.BatchNorm3d(64)
        self._initialize_conv_weights_to_one(self.conv1_x2)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1_x1 = self._make_layer(block, 64, layers[0], stride=1, kernel_size=3, branch='x1')
        self.layer2_x1 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=3, branch='x1')
        self.layer3_x1 = self._make_layer(block, 256, layers[2], stride=2, kernel_size=3, branch='x1')
        self.layer4_x1 = self._make_layer(block, 512, layers[3], stride=2, kernel_size=3, branch='x1')

        self.layer1_x2 = self._make_layer(block, 64, layers[0], stride=1, kernel_size=3, branch='x2')
        self.layer2_x2 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=3, branch='x2')
        self.layer3_x2 = self._make_layer(block, 256, layers[2], stride=2, kernel_size=3, branch='x2')
        self.layer4_x2 = self._make_layer(block, 512, layers[3], stride=2, kernel_size=3, branch='x2')

        self.inter_attention1 = InterAttention(64)
        self.inter_attention2 = InterAttention(128)
        self.inter_attention3 = InterAttention(256)
        self.inter_attention4 = InterAttention(512)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _initialize_conv_weights_to_one(self, conv_layer):
        with torch.no_grad():
            conv_layer.weight.fill_(1.0)
            if conv_layer.bias is not None:
                conv_layer.bias.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3, branch='x1'):
        downsample = None
        if branch == 'x1':
            inplanes = self.inplanes_x1
            conv_layer = 'x1'
        else:
            inplanes = self.inplanes_x2
            conv_layer = 'x2'

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
            if branch == 'x2':
                self._initialize_conv_weights_to_one(downsample[0])

        if branch == 'x1':
            layers = [block(inplanes, planes, stride, downsample, kernel_size)]
            self.inplanes_x1 = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes_x1, planes, kernel_size=3))
        else:
            layers = [block(inplanes, planes, stride, downsample, kernel_size)]
            self.inplanes_x2 = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes_x2, planes, kernel_size=3))

        layer = nn.Sequential(*layers)

        if branch == 'x2':
            for m in layer.modules():
                if isinstance(m, nn.Conv3d):
                    self._initialize_conv_weights_to_one(m)

        return layer

    def forward(self, x1, x2):
        x1 = self.conv1_x1(x1)
        x1 = self.bn1_x1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv1_x2(x2)
        x2 = self.bn1_x2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x1 = self.layer1_x1(x1)
        x2 = self.layer1_x2(x2)
        x1 = self.inter_attention1(x1, x2)

        x1 = self.layer2_x1(x1)
        x2 = self.layer2_x2(x2)
        x1 = self.inter_attention2(x1, x2)

        x1 = self.layer3_x1(x1)
        x2 = self.layer3_x2(x2)
        x1 = self.inter_attention3(x1, x2)

        x1 = self.layer4_x1(x1)
        x2 = self.layer4_x2(x2)
        x1 = self.inter_attention4(x1, x2)

        return x1, x2

class DualResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(DualResNet, self).__init__()
        self.stream = ResNetStream(block, layers)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)  # 512 * 2

    def forward(self, x1, x2):
        out1, out2 = self.stream(x1, x2)

        out1 = self.stream.avgpool(out1)
        out2 = self.stream.avgpool(out2)

        out1 = torch.flatten(out1, 1)
        out2 = torch.flatten(out2, 1)

        out = torch.cat((out1, out2), dim=1)

        out = self.fc(out)
        return out


def resnet34(**kwargs):
    model = DualResNet(BasicBlock3D, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == "__main__":
    # Initialize model
    model = resnet34(num_classes=10)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate random input data and move to GPU
    x1 = torch.randn(1, 1, 112, 128, 112).to(device)
    x2 = torch.randn(1, 1, 112, 128, 112).to(device)

    # Run a single forward pass to initialize the model and GPU
    with torch.no_grad():
        output = model(x1, x2)

    # Synchronize GPU and start timing
    torch.cuda.synchronize()
    start_time = time.time()

    # Run multiple forward passes to measure time
    with torch.no_grad():
        for _ in range(100):  # Run 100 forward passes
            output = model(x1, x2)
        torch.cuda.synchronize()

    # End timing
    end_time = time.time()

    # Calculate average runtime
    avg_time = (end_time - start_time) / 100
    print(f"Average inference time on GPU: {avg_time:.6f} seconds")

    # Print output shape
    print("Output shape:", output.shape)
