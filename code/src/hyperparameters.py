from resnet import ResNet
from residual_bloc import ResidualBlock
import torch.nn as nn
import torch
from data_prep import train_loader, device

num_classes  = 10
num_epochs = 20
batch_size = 16
learning_rate = 0.01

model = ResNet(
    ResidualBlock , [3, 4 , 6 , 3 ]
    ).to(device)

# loss and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr= learning_rate , weight_decay=0.001 , momentum=0.9)

# train the model 
total_step = len(train_loader)