import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

from model import DocumentCornerNet
from dataset import DocumentCornerDataset

def train(epochs=10, batch_size=16, learning_rate=1e-3, img_size=256, heatmap_size=64):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable cudnn benchmark for faster training if input sizes are constant
        torch.backends.cudnn.benchmark = True
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected. Using CPU. Training will be slow!")
        print("If you have an NVIDIA GPU, ensure you installed PyTorch with CUDA support.")
    # Load dataset
    full_dataset = DocumentCornerDataset('dataset/labels.json', 'dataset/images', img_size=img_size, heatmap_size=heatmap_size)
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # set num_workers=0 for windows
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = DocumentCornerNet(pretrained=True).to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Directory to save weights
    os.makedirs('weights', exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc="Training")
        
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_dataset)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'weights/best_model.pth')
            print("Saved new best model.")
            
    print("Training Complete!")

if __name__ == '__main__':
    train(epochs=15, batch_size=16)
