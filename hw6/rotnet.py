
import torch.nn as nn
import torch.nn.functional as F

class RotNet(nn.Module):

    def __init__(self, num_classes=4):
        super(RotNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = ResNetLayer(64, 64, 1)
        self.layer2 = ResNetLayer(64, 128, 2)
        self.layer3 = ResNetLayer(128, 256, 2)
        self.layer4 = ResNetLayer(256, 512, 2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=3)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetLayer, self).__init__()
        self.block1 = ResNetBlock(in_channels, out_channels, stride)
        self.block2 = ResNetBlock(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            shortcut = []
            shortcut.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            shortcut.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)
        return y
