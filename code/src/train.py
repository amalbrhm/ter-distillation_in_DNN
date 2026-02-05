import gc 
from data_prep import train_loader , valid_loader , device
from hyperparameters import num_epochs , model , optimizer , criterion
import torch

def train():
    # nombre de batches par epoch
    total_steps = len(train_loader)
    
    for epoch in range(num_epochs):
        for i , (images,labels) in enumerate(train_loader):
            # move tensors to the device
            
            images = images.to(device)
            labels = labels.to(device)
            
            # forward pass 
            outputs = model(images)
            loss = criterion(outputs , labels)
            
            # backward and optimize 
            optimizer.zero_grad()
            
            # calcul le gradient 
            loss.backward()
            
            # màj des parametres 
            optimizer.step()
            
            del images , labels , outputs
            torch.cuda.empty_cache()
            gc.collect()
            
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))
        
        # validation 
        with torch.no_grad() : 
            correct = 0 
            total = 0
            
            for images, labels in valid_loader:
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct+= (predicted == labels).sum().item()
                del images , labels , outputs
                
            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
            
    torch.save(model.state_dict(), "model.pth")
    print("Saved: model.pth")
    
if __name__ == "__main__":
    train()
