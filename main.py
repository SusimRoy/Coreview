import os
import math
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import VideoTokenMergingTransformer
from dataset import LVUDataset, BreakfastDataset, COINDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Video Token Merging Training')
    parser.add_argument('--dataset', type=str, default='LVU', choices=['LVU', 'Breakfast', 'COIN'],
                      help='Dataset to use for training')
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to dataset root directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def get_dataset(name, root_dir, split='train'):
    if name.upper() == 'LVU':
        return LVUDataset(root_dir, split=split, num_frames=60)
    elif name.upper() == 'BREAKFAST':
        return BreakfastDataset(root_dir, split=split, num_frames=64)
    elif name.upper() == 'COIN':
        return COINDataset(root_dir, split=split, num_frames=64)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def configure_optimizer(model, lr=0.001, weight_decay=0.01, num_epochs=70, warmup_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine scheduler with warmup
    num_epochs = 70
    warmup_epochs = 10
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs a single training epoch for the VideoTokenMergingTransformer.
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for videos, labels in progress_bar:
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        main_output, aux_loss = model(videos)
        
        main_loss = criterion(main_output, labels)
        
        # The final loss is the sum of the main task loss and the auxiliary loss
        # This allows the gradient to flow back through the auxiliary path
        total_epoch_loss = main_loss + aux_loss
        
        total_epoch_loss.backward()
        optimizer.step()
        
        total_loss += total_epoch_loss.item()
        progress_bar.set_postfix({
            "Loss": f"{total_epoch_loss.item():.4f}",
            "Main": f"{main_loss.item():.4f}",
            "Aux": f"{aux_loss.item():.4f}"
        })
        
    avg_loss = total_loss / len(dataloader)
    print(f"Training Epoch Finished. Average Loss: {avg_loss:.4f}")

def evaluate(model, dataloader, criterion, device):
    """
    Runs evaluation. Note that in eval mode, the model should not return an aux_loss.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            
            # In eval mode, aux_loss is 0 and the auxiliary path is skipped
            main_output, _ = model(videos)
            
            loss = criterion(main_output, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(main_output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100
    print(f"Evaluation Finished. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def main():
    args = parse_args()
    print(f"Using device: {args.device}")
    
    # Create datasets and dataloaders
    train_dataset = get_dataset(args.dataset, args.data_path, split='train')
    val_dataset = get_dataset(args.dataset, args.data_path, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)
    
    num_classes = {'LVU': 9, 'Breakfast': 10, 'COIN': 180}[args.dataset.upper()]
    num_frames = 60 if args.dataset.upper() == 'LVU' else 64
    patch_dim = 1024  
    
    model = VideoTokenMergingTransformer(
        num_classes=num_classes,
        num_frames=num_frames,
        patch_dim=patch_dim,
        num_vtm_blocks=3,
        num_heads=8,
        dataset=args.dataset
    ).to(args.device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = configure_optimizer(
        model, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Train
        train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{args.dataset.lower()}.pth')
    
    print(f"\n✅ Training complete. Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
