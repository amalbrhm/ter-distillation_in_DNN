import torch.nn as nn 

class ResNet(nn.Module):
    def __init__(self, block , layers , num_classes = 10):
        
        super(ResNet , self).__init__()
        # nbr de canaux courants à l'entée du prochain stage, on commence à 64 pcq la premiere conv produit 64 canaux
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 7 , stride = 2 , padding = 3 ),
            # stabilise l'entrainement
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # appel aux blocs sequentiels de plusieurs ResidualBlock
        self.maxpool = nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1)
        self.layer0 = self._make_layer(block , 64 , layers[0] , stride = 1)
        self.layer1 = self._make_layer(block , 128 , layers[1] , stride = 2)
        self.layer2 = self._make_layer(block , 256 , layers[2] , stride = 2)
        self.layer3 = self._make_layer(block , 512 , layers[3] , stride = 2)
        # moyenne sur fenetre de 7 x 7
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.fc = nn.Linear(512 , num_classes)
    
    def _make_layer(self , block , planes , blocks , stride = 1 ): # planes ( canaux de sortie du stage)
        # par defaut la connexion residuelle est l'identité ( pas de projection )
        downsample = None
        if stride != 1 or self.inplanes != planes :
            # Projection 1×1 qui :ajuste les canaux inplanes -> planes , ajuste la résolution si stride=2 , BatchNorm derrière (standard ResNet)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes , planes , kernel_size = 1 , stride = stride),
                nn.BatchNorm2d(planes),
            )
        
        # liste des blocs qui formeront le stage     
        layers = []
        # ajout du premier bloc
        layers.append(block(self.inplanes , planes , stride , downsample))
        # Mise à jour : après le premier bloc, la sortie a désormais planes canaux
        self.inplanes = planes 
        
        #ajout des blocs restants 
        for i in range(1 , blocks):
            layers.append(block(self.inplanes , planes))
        
        return nn.Sequential(*layers)
    
    def forward(self , x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        
        x = self.avgpool(x)
        # flatten transforme (batch, 512, 1, 1) en (batch, 512)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
        