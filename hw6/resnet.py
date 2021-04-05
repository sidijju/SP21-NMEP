
import torch.nn as nn
import torch.nn.functional as F

class ResNet:

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.conv1 = nn.Conv2D(3, 64, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=2)
        self.layer1 = new_block(64, 64, 1)
        self.layer2 = new_block(128, 128, 1)
        self.layer3 = new_block(256, 256, 1)
        self.layer4 = new_block(512, 512, 1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, 1000)


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
        x = self.fc(x)

class ResNetLayer:
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetLayer, self).__init__()
        self.block1 = ResNetBlock(in_channels, in_channels, stride)
        self.block2 = ResNetBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

class ResNetBlock:
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, in_channels, out_channels, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, out_channels, out_channels, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            shortcut = []
            shortcut += nn.Conv2d(1, in_channels, out_channels, stride=stride)
            shortcut += nn.BatchNorm(out_channels)
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
