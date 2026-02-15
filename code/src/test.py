import gc 
from data_prep import  device , data_loader
from hyperparameters import  model ,model_name
import torch

def test(data_dir, model_name):
    
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        test_loader = data_loader(data_dir=data_dir ,batch_size=64 ,  test=True)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
 
       
import argparse
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="././data")
    parser.add_argument("--model_name", type=str , default="resnet_d20_w16.pth")

    args = parser.parse_args()
    
    model_name = "././models/" + args.model_name
    
    test(args.data_dir, model_name)