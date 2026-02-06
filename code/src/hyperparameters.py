

import torch.nn as nn
import torch
from data_prep import train_loader, device
from resnet_cifar import resnet20_cifar
from resnet2 import resnet20 
num_classes  = 10
num_epochs = 20
batch_size = 16
learning_rate = 0.01

# resnet-10 [ 1, 1, 1, 1]
# resnet-18 [ 2, 2, 2 ,2]
# resnet-34 [ 3, 4, 6, 3] 6n + 2 = 32 => n = 5


model = resnet20().to(device)
model_name = "resnet20_w16"

# loss and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr= learning_rate , weight_decay=0.001 , momentum=0.9)

# train the model 
total_step = len(train_loader)