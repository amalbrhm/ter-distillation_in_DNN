import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlockCIFAR(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, n, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, n, stride=1)  # 32x32
        self.layer2 = self._make_layer(32, n, stride=2)  # 16x16
        self.layer3 = self._make_layer(64, n, stride=2)  # 8x8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlockCIFAR(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockCIFAR(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet20_cifar(num_classes=10):
    return ResNetCIFAR(n=3, num_classes=num_classes)  # 6*3+2 = 20


def resnet32_cifar(num_classes=10):
    return ResNetCIFAR(n=5, num_classes=num_classes)  # 6*5+2 = 32
