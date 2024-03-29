import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(
                Conv2d(
                    inchannel,
                    outchannel,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 4, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 8, 2, stride=2)

        # Uncomment for bigger model
        # self.layer3 = self.make_layer(ResidualBlock, 16, 2, stride=2)
        # self.layer4 = self.make_layer(ResidualBlock, 32, 2, stride=1)
        self.maxpool = MaxPool2d(4)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # Uncomment for bigger model
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.maxpool(x)

        return x


def ResNet18():
    return ResNet(ResidualBlock)
