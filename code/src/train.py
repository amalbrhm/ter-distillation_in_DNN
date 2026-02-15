import gc 
from data_prep import  data_loader
from hyperparameters import num_epochs , model , optimizer , criterion , model_name
import torch
import os
import argparse

import time, json, csv, os
from datetime import datetime
import torch
import time
from datetime import datetime
import os

def get_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train(data_dir):
    
    train_loader , valid_loader = data_loader(data_dir , batch_size= 64 , test=False)
    # nombre de batches par epoch
    total_steps = len(train_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    start_time = time.perf_counter()
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
            
    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Total training time: {total_time/60:.2f} minutes")
    
    total_params, trainable_params = get_model_stats(model)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Création dossier logs
    os.makedirs("logs", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{run_id}.txt"

    with open(log_path, "w") as f:
        f.write("===== TRAINING SUMMARY =====\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")

        f.write("MODEL CHARACTERISTICS\n")
        f.write(f"Model class: {model.__class__.__name__}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n\n")

        f.write("TRAINING INFO\n")
        f.write(f"Device: {device}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Total training time (seconds): {total_time:.4f}\n")
        f.write(f"Total training time (minutes): {total_time/60:.4f}\n")

    print(f"\nTraining summary saved to {log_path}")
    
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Saved: {save_path}")
    
    with open(os.path.join(save_dir, f"{model_name}.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(str(model))

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="././data")
    args = parser.parse_args()
    train(args.data_dir)
