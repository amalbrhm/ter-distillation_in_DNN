

import torch.nn as nn
import torch
from data_prep import    data_loader
from resnet import resnet20 , resnet32 , resnet56
from vgg import VGG_CIFAR
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes  = 10
num_epochs = 5
batch_size = 16 # ou 64
learning_rate = 0.01

# resnet-10 [ 1, 1, 1, 1]
# resnet-18 [ 2, 2, 2 ,2]
# resnet-34 [ 3, 4, 6, 3] 6n + 2 = 32 => n = 5


model = resnet20(width=32).to(device)
model_name =  "resn_20_32"
print(model_name)
# loss and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr= learning_rate , weight_decay=0.001 , momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)