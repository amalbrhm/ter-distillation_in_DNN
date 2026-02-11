import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_CIFAR(nn.Module):
    """
    Implémentation de VGG pour CIFAR-10 (images 32x32).

    Paramètre :
    ----------------------
    d : profondeur (nombre de couches convolutionnelles)

    L’architecture est organisée en 5 blocs séparés par des MaxPool.

    Exemple :
    ----------
    d = 11  → proche d’un VGG11
    d = 16  → proche d’un VGG16

    Structure générale :
        [Conv-BN-ReLU] x k1 → MaxPool
        [Conv-BN-ReLU] x k2 → MaxPool
        ...
        5 blocs au total
        Puis Global Average Pooling
        Puis Linear
    """

    def __init__(self, d: int = 11, num_classes: int = 10, in_channels: int = 3, base_channels: int = 64):

        super().__init__()

        # On répartit d convolutions sur 5 blocs
        # Exemple : d = 11 → [3,2,2,2,2]
        convs_per_block = [d // 5] * 5
        print(convs_per_block)
        remainder = d % 5
        for i in range(remainder):
            convs_per_block[i] += 1
        print(convs_per_block)
        # -------------------------------------------------------
        # tailles de canaux par bloc
        # -------------------------------------------------------

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8
        ]

        layers = []
        c_in = in_channels

        # -------------------------------------------------------
        # 3Construction des 5 blocs convolutionnels
        # -------------------------------------------------------

        for block_idx in range(5):

            c_out = channels[block_idx]

            # Ajout des convolutions du bloc
            for _ in range(convs_per_block[block_idx]):
                layers.append(
                    nn.Conv2d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    )
                )
                layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU(inplace=True))

                c_in = c_out  # mise à jour du nombre de canaux

            # Après chaque bloc -> réduction spatiale
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)

        # -------------------------------------------------------
        #  Tete de classification adaptée CIFAR
        # -------------------------------------------------------

        # Après 5 MaxPool sur 32x32 :
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # On obtient un tenseur (B, C, 1, 1)

        self.classifier = nn.Linear(c_in, num_classes)

        self._init_weights()

    # -----------------------------------------------------------
    # Initialisation des poids (He initialization)
    # -----------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # Kaiming He et al., ICCV 2015 sans cette correction, les activations explosent ou disparaissent lorsque la profondeur augmente.
            elif isinstance(m, nn.BatchNorm2d):  # Ioffe & Szegedy, ICML 2015
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): # Krizhevsky et al., 2012 pour éviter des logits trop grands au départ
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------

    def forward(self, x):

        # Passage dans les blocs convolutionnels
        x = self.features(x)

        # Global Average Pooling
        # (B, C, H, W) → (B, C, 1, 1)
        x = F.adaptive_avg_pool2d(x, 1)

        # Flatten : (B, C, 1, 1) → (B, C)
        x = torch.flatten(x, 1)

        # Classification finale
        x = self.classifier(x)

        return x


# ---------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------

if __name__ == "__main__":

    model = VGG_CIFAR(d=11)

    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    print("Sortie :", y.shape)  # attendu : torch.Size([4, 10])
    print(x)
    print(y)
