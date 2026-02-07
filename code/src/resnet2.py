"""
 (profondeur = 6n + 2 et largeur configurable)

Idée générale :
- Entrées type CIFAR-10/100 : (N, 3, 32, 32)
- 3 "stages" :
    width  -> 2*width -> 4*width
- Pooling global avec AdaptiveAvgPool2d(1)

Exemples calcul de n :
- ResNet-20 : n=3
- ResNet-32 : n=5
- ResNet-56 : n=9

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1) Bloc résiduel "BasicBlock" (2 convolutions 3x3)
class BasicBlock(nn.Module):
    """
    - 2 convolutions 3x3
    - une connexion résiduelle (skip connection) : out = F(x) + shortcut(x)

    expansion = 1 pour la sortie a "planes" canaux.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        # 1ère convolution 
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # 2ème convolution 
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Par défaut shortcut(x) = x
        self.shortcut = nn.Identity()

        # Si la résolution change (stride != 1) OU si le nb de canaux change,
        # on doit projeter x pour pouvoir faire l'addition out + x sinon pas les memes dimensions
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                # Conv 1x1 : ajuste canaux et résolution via stride
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chemin principal F(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # Ajout résiduel (skip connection)
        out = out + self.shortcut(x)

        # Activation finale
        out = F.relu(out, inplace=True)
        return out


# 2) Réseau ResNetCIFAR : 3 stages (width, 2*width, 4*width)
class ResNetCIFAR(nn.Module):
    """
    - profondeur = 6n + 2 
      (3 stages, chaque stage a n blocs, chaque bloc a 2 conv => 6n conv + conv1 + fc)

    Paramètres :
    - n : nombre de blocs par stage (ex: n=3 => ResNet-20)
    - width : largeur de base 
    - num_classes : nb de classes ( =10 ici)
    """
    def __init__(self, n: int, width: int = 16, num_classes: int = 10, block=BasicBlock):
        super().__init__()
        self.block = block

        # in_planes suit le nombre de canaux "courant" à l'entrée du prochain stage
        self.in_planes = width

        # Stem CIFAR : Conv 3x3 stride 1, pas de maxpool
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # Stages :
        # - layer1 : conserve 32x32, nb canaux = width
        # - layer2 : downsample -> 16x16, nb canaux = 2*width
        # - layer3 : downsample -> 8x8,  nb canaux = 4*width
        self.layer1 = self._make_layer(planes=width, blocks=n, stride=1)
        self.layer2 = self._make_layer(planes=2 * width, blocks=n, stride=2)
        self.layer3 = self._make_layer(planes=4 * width, blocks=n, stride=2)

        # Pooling global : (N, C, H, W) -> (N, C, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Couche de classification : vecteur de taille C -> num_classes
        self.fc = nn.Linear(4 * width * self.block.expansion, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        """
        Construit un stage composé de `blocks` blocs résiduels.
        - Le 1er bloc peut downsample via `stride` (si stride=2)
        - Les suivants gardent stride=1
        """
        layers = []

        # 1er bloc du stage 
        layers.append(self.block(self.in_planes, planes, stride))
        self.in_planes = planes * self.block.expansion

        # Blocs restants 
        for _ in range(1, blocks):
            layers.append(self.block(self.in_planes, planes, stride=1))
            self.in_planes = planes * self.block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # 3 stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Pooling global + flatten
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        # Classification
        out = self.fc(out)
        return out


# 3) Fonctions "builders" pour créer des modèles standards

def resnet20(width: int = 16, num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20 CIFAR : n=3 car 6*3 + 2 = 20"""
    return ResNetCIFAR(n=3, width=width, num_classes=num_classes, block=BasicBlock)

def resnet32(width: int = 16, num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-32 CIFAR : n=5 car 6*5 + 2 = 32"""
    return ResNetCIFAR(n=5, width=width, num_classes=num_classes, block=BasicBlock)

def resnet56(width: int = 16, num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-56 CIFAR : n=9 car 6*9 + 2 = 56"""
    return ResNetCIFAR(n=9, width=width, num_classes=num_classes, block=BasicBlock)


def _test():
    net = resnet20(width=16, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print("Output shape:", y.shape)  # attendu : (2, 10)

if __name__ == "__main__":
    _test()
