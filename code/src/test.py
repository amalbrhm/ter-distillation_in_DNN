import gc 
from data_prep import test_loader , device
from hyperparameters import  model 
import torch
from train import train

def test():
    
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
 
       

        
test()