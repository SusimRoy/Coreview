import os
import math
import ot
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from matplotlib.pylab import size
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.breakfast_dataset import CustomDataset
from models.model_lvu import VideoTokenMergingTransformer
import matplotlib.pyplot as plt
import numpy as np
torch.inverse(torch.ones((0, 0), device="cuda:0"))

def parse_args():
    parser = argparse.ArgumentParser(description='Video Token Merging Training')
    parser.add_argument('--dataset', type=str, default='Breakfast', choices=['LVU', 'Breakfast', 'COIN'],
                      help='Dataset to use for training')
    parser.add_argument('--save_dir', type=str, default='/data/susimmuk/long-video/', help='Directory to save results')
    # parser.add_argument('--data-path', type=str, required=True,
    parser.add_argument('--l_secs', default=64, type=int, help='l_secs')

    #                   help='Path to dataset root directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

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
    for id, video_feat, labels in progress_bar:
        video_feat, labels = video_feat.to(device), labels.to(device)
        # print(video_feat.shape)
        # print(labels.shape)
        
        optimizer.zero_grad()
        
        main_output,aux_output = model(video_feat)

        aux_loss = criterion(aux_output, labels) if aux_output is not None else 0.0
        
        main_loss = criterion(main_output, labels)
        
        # total_epoch_loss = 0.9*main_loss + 0.1*aux_loss
        # total_epoch_loss = 0.9*main_loss + 0.1*aux_loss
        total_epoch_loss = 0.9*main_loss + 0.099*aux_loss + 5e-3*ot_loss.mean()
        # total_epoch_loss = main_loss
        # if iter%8 == 0:
        total_epoch_loss.backward()
        optimizer.step()
        
        total_loss += total_epoch_loss.item()
        progress_bar.set_postfix({
            "Loss": f"{total_epoch_loss.item():.4f}",
            "Main": f"{0.9*main_loss.item():.4f}",
            "Aux": f"{0.1*aux_loss.item():.4f}",
            # "OT": f"{1e-3*ot_loss.mean().item():.4f}"
        })
        
    avg_loss = total_loss / len(dataloader)
    print(f"Training Epoch Finished. Average Loss: {avg_loss:.4f}")
    return avg_loss  # Return average loss for plotting

def evaluate(model, dataloader, criterion, device):
    """
    Runs evaluation. Note that in eval mode, the model should not return an aux_loss.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for id, videos, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            
            # In eval mode, aux_loss is 0 and the auxiliary path is skipped
            main_output = model(videos)
            
            loss = criterion(main_output, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(main_output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100
    print(f"Evaluation Finished. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def main():
    args = parse_args()
    print(f"Using device: {args.device}")

    os.makedirs(args.save_dir, exist_ok=True)
    trainset = CustomDataset(args=args, split='train')
    valset = CustomDataset(args=args, split='test')

    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    num_classes = {'LVU': 9, 'Breakfast': 10, 'COIN': 180}[args.dataset]
    patch_dim = 1024
    if args.dataset.upper() == 'LVU':
        num_frames = 60
        num_tokens = 16 * 16 * num_frames  # ViT-L: 16x16 patches
    else:
        num_frames = 64
        num_tokens = 7 * 7 * num_frames    # Swin-B: 7x7 patches

    model = VideoTokenMergingTransformer(
        num_classes=num_classes,
        num_tokens=num_tokens,
        patch_dim=patch_dim,
        num_vtm_blocks=3,
        num_heads=8
    )

    print(model)
    
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = configure_optimizer(
        model, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs
    )
    
    best_acc = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    gamma_discrete_values = []
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Get gamma info
        # if hasattr(model, 'module'):
        #     gamma_info = model.module.vtm_blocks[0].get_gamma_info()
        # else:
        #     gamma_info = model.vtm_blocks[0].get_gamma_info()
        gamma_discrete_values.append(6)
        # print(f"Gamma Info: Continuous={gamma_info['gamma_continuous']:.2f}, Discrete={gamma_info['gamma_discrete']}")
        
        # Training
        train_loss = train_one_epoch(model, trainloader, optimizer, criterion, args.device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_acc = evaluate(model, valloader, criterion, args.device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save the underlying model state dict when using DataParallel
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), f'{args.save_dir}/best_model_original_{args.dataset.lower()}.pth')
    
    print(f"\n✅ Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    # Save training curves

if __name__ == "__main__":
    main()
