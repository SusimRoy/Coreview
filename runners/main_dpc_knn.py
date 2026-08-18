# import os
# import math
# import ot
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import argparse
# from matplotlib.pylab import size
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from datasets.breakfast_dataset import CustomDataset
# from models.model_select_merge import HybridVideoTokenMergingTransformer
# import matplotlib.pyplot as plt
# import numpy as np
# from models.visualization import visualize_token_score_distributions
# torch.inverse(torch.ones((0, 0), device="cuda:0"))

# def parse_args():
#     parser = argparse.ArgumentParser(description='DPC-KNN Video Token Merging Training')
#     parser.add_argument('--dataset', type=str, default='LVU', choices=['LVU', 'Breakfast', 'COIN'],
#                       help='Dataset to use for training')
#     parser.add_argument('--l_secs', default=64, type=int, help='l_secs')
    
#     # DPC-KNN specific parameters
#     parser.add_argument('--k_neighbors', type=int, default=2, help='Number of neighbors for DPC-KNN')
#     parser.add_argument('--target_ratio', type=float, default=0.8, help='Target ratio of tokens to select')

#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=70)
#     parser.add_argument('--lr', type=float, default=0.0001)
#     parser.add_argument('--weight-decay', type=float, default=0.05)
#     parser.add_argument('--num-workers', type=int, default=8)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
#     return parser.parse_args()


# def configure_optimizer(model, lr=0.001, weight_decay=0.01, num_epochs=70, warmup_epochs=10):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
#     # Cosine scheduler with warmup
#     def lr_lambda(epoch):
#         if epoch < warmup_epochs:
#             return epoch / warmup_epochs
#         return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
#     return optimizer, scheduler


# def save_training_curves(train_losses, val_losses, val_accuracies, k_neighbors_values, target_ratio_values, save_dir='./plots', dataset_name='model'):
#     """
#     Save training loss and validation accuracy curves for DPC-KNN model
    
#     Args:
#         train_losses: List of training losses per epoch
#         val_losses: List of validation losses per epoch  
#         val_accuracies: List of validation accuracies per epoch
#         k_neighbors_values: List of k_neighbors values per epoch (constant for DPC-KNN)
#         target_ratio_values: List of target_ratio values per epoch (constant for DPC-KNN)
#         save_dir: Directory to save plots
#         dataset_name: Name for file prefix
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     epochs = range(1, len(train_losses) + 1)
    
#     # Create subplots
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle(f'DPC-KNN Training Curves - {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
#     # Plot 1: Training and Validation Loss
#     ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
#     ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.set_title('Training & Validation Loss')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#     ax1.set_xlim(1, len(epochs))
    
#     # Plot 2: Validation Accuracy
#     ax2.plot(epochs, val_accuracies, 'g-^', label='Validation Accuracy', linewidth=2, markersize=4)
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy (%)')
#     ax2.set_title('Validation Accuracy')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#     ax2.set_xlim(1, len(epochs))
    
#     # Find best accuracy
#     best_epoch = np.argmax(val_accuracies) + 1
#     best_acc = max(val_accuracies)
#     ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
#     ax2.annotate(f'Best: {best_acc:.2f}% (Epoch {best_epoch})', 
#                 xy=(best_epoch, best_acc), xytext=(best_epoch+2, best_acc-5),
#                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
#                 fontsize=10, color='red')
    
#     # Plot 3: DPC-KNN Parameters (constant lines)
#     ax3.axhline(y=k_neighbors_values[0], color='m', linewidth=2, label=f'k_neighbors = {k_neighbors_values[0]}')
#     ax3.axhline(y=target_ratio_values[0]*10, color='c', linewidth=2, label=f'target_ratio = {target_ratio_values[0]} (×10)')
#     ax3.set_xlabel('Epoch')
#     ax3.set_ylabel('Parameter Value')
#     ax3.set_title('DPC-KNN Parameters (Fixed)')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
#     ax3.set_xlim(1, len(epochs))
    
#     # Plot 4: Loss vs Accuracy Correlation
#     ax4.scatter(val_losses, val_accuracies, c=epochs, cmap='viridis', s=50, alpha=0.7)
#     ax4.set_xlabel('Validation Loss')
#     ax4.set_ylabel('Validation Accuracy (%)')
#     ax4.set_title('Loss vs Accuracy Correlation')
#     colorbar = plt.colorbar(ax4.collections[0], ax=ax4)
#     colorbar.set_label('Epoch')
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save the combined plot
#     plot_path = os.path.join(save_dir, f'{dataset_name}_dpc_knn_training_curves.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
#     plt.close()
    
#     # Save individual plots for detailed analysis
#     # Training Loss only
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Training Loss', fontsize=12)
#     plt.title(f'DPC-KNN Training Loss - {dataset_name.upper()}', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'{dataset_name}_dpc_knn_train_loss.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Validation Accuracy only
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, val_accuracies, 'g-^', linewidth=2, markersize=4)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Validation Accuracy (%)', fontsize=12)
#     plt.title(f'DPC-KNN Validation Accuracy - {dataset_name.upper()}', fontsize=14, fontweight='bold')
#     plt.axhline(y=best_acc, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'{dataset_name}_dpc_knn_val_accuracy.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Save metrics to text file
#     metrics_path = os.path.join(save_dir, f'{dataset_name}_dpc_knn_training_metrics.txt')
#     with open(metrics_path, 'w') as f:
#         f.write(f"DPC-KNN Training Metrics - {dataset_name.upper()}\n")
#         f.write("="*60 + "\n\n")
#         f.write("DPC-KNN Configuration:\n")
#         f.write(f"  k_neighbors: {k_neighbors_values[0]}\n")
#         f.write(f"  target_ratio: {target_ratio_values[0]}\n\n")
#         f.write(f"Total Epochs: {len(epochs)}\n")
#         f.write(f"Best Validation Accuracy: {best_acc:.4f}% (Epoch {best_epoch})\n")
#         f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
#         f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n")
#         f.write(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}%\n\n")
        
#         f.write("Epoch-wise Details:\n")
#         f.write("-"*80 + "\n")
#         f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\tk_neighbors\ttarget_ratio\n")
#         f.write("-"*80 + "\n")
#         for i, epoch in enumerate(epochs):
#             f.write(f"{epoch}\t{train_losses[i]:.6f}\t{val_losses[i]:.6f}\t{val_accuracies[i]:.4f}\t{k_neighbors_values[i]}\t{target_ratio_values[i]}\n")
    
#     print(f"\n📊 DPC-KNN Training curves saved to: {save_dir}")
#     print(f"   - Combined plot: {dataset_name}_dpc_knn_training_curves.png")
#     print(f"   - Individual plots: {dataset_name}_dpc_knn_train_loss.png, {dataset_name}_dpc_knn_val_accuracy.png")
#     print(f"   - Metrics file: {dataset_name}_dpc_knn_training_metrics.txt")
#     print(f"   - Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")

# def check_gradients(model):
#     """Helper function to check gradient statistics"""
#     total_norm = 0
#     parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
#     for p in parameters:
#         param_norm = p.grad.detach().data.norm(2)
#         total_norm += param_norm.item() ** 2
#     total_norm = total_norm ** 0.5
#     return total_norm

# def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
#     """
#     Runs a single training epoch for the DPC-KNN VideoTokenMergingTransformer.
#     """
#     model.train()
#     total_loss = 0.0
#     torch.backends.cuda.preferred_linalg_library()
#     progress_bar = tqdm(dataloader, desc="Training", leave=False)
#     i = 0
#     grad_norms = []
#     for (id, video_feat, labels) in progress_bar:
#         video_feat, labels = video_feat.to(device), labels.to(device)
        
#         optimizer.zero_grad()
        
#         main_output, token_list,_,scores_list= model(video_feat, epoch, id)
#                         # aux_loss = criterion(aux_output, labels) if aux_output is not None else 0.0
#         loss1 = criterion(main_output, labels)
#         # loss1.backward()
#         loss2 = 1e-3*OT_Loss(token_list, scores_list).mean()
#         # loss2.backward()
#         main_loss = loss1 + loss2
#         # main_loss = criterion(main_output, labels) + 1e-3*OT_Loss(token_list, scores_list).mean()
        
#         main_loss.backward()
#         # grad_norm = check_gradients(model)
#         # grad_norms.append(grad_norm)
        
#         # # Warning for exploding gradients
#         # if grad_norm > 10.0:
#         #     print(f"\n⚠️ Warning: Gradient norm explosion detected: {grad_norm:.2f}")
        
#         # # Warning for vanishing gradients
#         # if grad_norm < 1e-3:
#         #     print(f"\n⚠️ Warning: Possible vanishing gradients: {grad_norm:.2f}")
      
#         optimizer.step()
        
#         total_loss += main_loss.item()
#         progress_bar.set_postfix({
#             "Loss": f"{main_loss.item():.4f}",
#             "Main": f"{main_loss.item():.4f}",
#             # "Aux": f"{aux_loss.item():.4f}"
#         })
#         i+=1
        
#     avg_loss = total_loss / len(dataloader)
#     print(f"Training Epoch Finished. Average Loss: {avg_loss:.4f}")
#     return avg_loss

# def OT_Loss(token_list, scores_list):
#     loss = 0
#     for i in range(len(token_list)-1):
#         if i == len(token_list)-2:
#             loss += compute_OT(token_list[0], token_list[i+1])
#         else:
#             loss += compute_OT(token_list[0], token_list[i+1], scores_list[0], scores_list[i+1])
#     return loss
# def compute_OT(src, tgt, s_src=None, s_tgt=None):
#         C1 = torch.cdist(src, src).pow(2)
#         C2 = torch.cdist(tgt, tgt).pow(2)
#         M = torch.cdist(src, tgt).pow(2)
#         if s_src is not None:
#             G = ot.solve_gromov_batch(C1, C2, M=M, alpha=0.5, a=s_src, b=s_tgt)
#         else:
#             G = ot.solve_gromov_batch(C1, C2, M=M, alpha=0.5)
#         return G.value.mean()

# def evaluate(model, dataloader, criterion, device, epoch):
#     """
#     Runs evaluation. Note that in eval mode, the model should not return an aux_loss.
#     """
#     model.eval()
#     total_loss = 0.0
#     correct_predictions = 0
#     total_samples = 0
    
#     with torch.no_grad():
#         for id_num, (id, videos, labels) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
#             videos, labels = videos.to(device), labels.to(device)
            
#             # In eval mode, aux_loss is 0 and the auxiliary path is skipped
#             # p = torch.linalg.inv(torch.ones((0, 0), device="cuda:0"))
#             main_output,_, = model(videos, epoch, id_num)
            
#             loss = criterion(main_output, labels)
#             total_loss += loss.item()
            
#             _, predicted = torch.max(main_output, 1)
#             correct_predictions += (predicted == labels).sum().item()
#             total_samples += labels.size(0)
            
#     avg_loss = total_loss / len(dataloader)
#     accuracy = (correct_predictions / total_samples) * 100
#     print(f"Evaluation Finished. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#     return avg_loss, accuracy


# def main():
#     args = parse_args() 
#     print(f"Using device: {args.device}")
    
#     trainset = CustomDataset(args=args, split='train')
#     valset = CustomDataset(args=args, split='test')

#     trainloader = torch.utils.data.DataLoader(
#             trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     valloader = torch.utils.data.DataLoader(
#             valset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
#     num_classes = {'LVU': 9, 'Breakfast': 10, 'COIN': 180}[args.dataset]
#     patch_dim = 1024
#     if args.dataset.upper() == 'LVU':
#         num_frames = 60
#         num_tokens = 16 * 16 * num_frames  # ViT-L: 16x16 patches
#     else:
#         num_frames = 64
#         num_tokens = 7 * 7 * num_frames    # Swin-B: 7x7 patches

#     # Create DPC-KNN model instead of original model
#     model = HybridVideoTokenMergingTransformer(
#         num_classes=num_classes,
#         num_tokens=num_tokens,
#         patch_dim=patch_dim,
#         num_vtm_blocks=3,
#         num_heads=8,
#         k_neighbors=args.k_neighbors,
#         target_ratio=args.target_ratio
#     )
    
#     # Use DataParallel for multi-GPU training
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
#         model = nn.DataParallel(model)
    
#     model = model.to(args.device)

    
#     # Print model info
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer, scheduler = configure_optimizer(
#         model, 
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         num_epochs=args.epochs
#     )

#     if args.evaluate:
#         print('==> Evaluating model..')
#         checkpoint = torch.load(f'/home/csgrad/susimmuk/long-video/runners/best_model_breakfast_dpc_knn_breakfast_92.1127.pth', map_location=args.device)
#         # model.module.load_state_dict(checkpoint)
#         model.load_state_dict(checkpoint)
#         print('Model loaded.')
#         evaluate(model, valloader, criterion, args.device, 0)
#         # Use the underlying model (not DataParallel wrapper) for visualization
#         # model_for_vis = model.module if hasattr(model, 'module') else model
#         # block_scores = visualize_token_score_distributions(model_for_vis, valloader, args.device)
#         return
    
#     best_acc = 0
    
#     # Lists to store metrics for plotting
#     train_losses = []
#     val_losses = []
#     val_accuracies = []
#     k_neighbors_values = []
#     target_ratio_values = []
    
#     for epoch in range(args.epochs):
#         print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
#         # Store DPC-KNN parameters (constant throughout training)
#         k_neighbors_values.append(args.k_neighbors)
#         target_ratio_values.append(args.target_ratio)
        
#         # Training
#         train_loss = train_one_epoch(model, trainloader, optimizer, criterion, args.device, epoch)
#         train_losses.append(train_loss)
        
#         # Validation
#         val_loss, val_acc = evaluate(model, valloader, criterion, args.device, epoch)
#         val_losses.append(val_loss)
#         val_accuracies.append(val_acc)
        
#         scheduler.step()
        
#         if val_acc > best_acc:
#             best_acc = val_acc
#             # Save the underlying model state dict when using DataParallel
#             model_to_save = model.module if hasattr(model, 'module') else model
    
#     torch.save(model_to_save.state_dict(), f'best_model_{args.dataset.lower()}_dpc_knn_breakfast_{best_acc:.4f}.pth')
    
#     print(f"\n✅ DPC-KNN Training complete. Best validation accuracy: {best_acc:.2f}%")
    
#     # Save training curves
#     save_training_curves(
#         train_losses=train_losses,
#         val_losses=val_losses, 
#         val_accuracies=val_accuracies,
#         k_neighbors_values=k_neighbors_values,
#         target_ratio_values=target_ratio_values,
#         save_dir='./training_plots',
#         dataset_name=args.dataset.lower()
#     )


# if __name__ == "__main__":
#     main()


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
from models.visualization import visualize_token_score_distributions
torch.inverse(torch.ones((0, 0), device="cuda:0"))

def parse_args():
    parser = argparse.ArgumentParser(description='Video Token Merging Training')
    parser.add_argument('--dataset', type=str, default='LVU', choices=['LVU', 'Breakfast', 'COIN'],
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

def save_training_curves(train_losses, val_losses, val_accuracies, gamma_values, save_dir='./plots', dataset_name='model'):
    """
    Save training loss and validation accuracy curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch  
        val_accuracies: List of validation accuracies per epoch
        gamma_values: List of gamma values per epoch
        save_dir: Directory to save plots
        dataset_name: Name for file prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves - {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    
    # Plot 2: Validation Accuracy
    ax2.plot(epochs, val_accuracies, 'g-^', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    
    # Find best accuracy
    best_epoch = np.argmax(val_accuracies) + 1
    best_acc = max(val_accuracies)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_acc:.2f}% (Epoch {best_epoch})', 
                xy=(best_epoch, best_acc), xytext=(best_epoch+2, best_acc-5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    # Plot 3: Gamma Evolution
    ax3.plot(epochs, gamma_values, 'm-o', label='Learnable Gamma', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gamma Value')
    ax3.set_title('Gamma Evolution During Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(epochs))
    
    # Plot 4: Loss vs Accuracy Correlation
    ax4.scatter(val_losses, val_accuracies, c=epochs, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Validation Loss')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_title('Loss vs Accuracy Correlation')
    colorbar = plt.colorbar(ax4.collections[0], ax=ax4)
    colorbar.set_label('Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = os.path.join(save_dir, f'{dataset_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    plt.close()
    
    # Save individual plots for detailed analysis
    # Training Loss only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title(f'Training Loss - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_train_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Validation Accuracy only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracies, 'g-^', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title(f'Validation Accuracy - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.axhline(y=best_acc, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_val_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to text file
    metrics_path = os.path.join(save_dir, f'{dataset_name}_training_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Training Metrics - {dataset_name.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Best Validation Accuracy: {best_acc:.4f}% (Epoch {best_epoch})\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n")
        f.write(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}%\n")
        f.write(f"Final Gamma Value: {gamma_values[-1]}\n\n")
        
        f.write("Epoch-wise Details:\n")
        f.write("-"*80 + "\n")
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\tGamma\n")
        f.write("-"*80 + "\n")
        for i, epoch in enumerate(epochs):
            f.write(f"{epoch}\t{train_losses[i]:.6f}\t{val_losses[i]:.6f}\t{val_accuracies[i]:.4f}\t{gamma_values[i]}\n")
    
    print(f"\n📊 Training curves saved to: {save_dir}")
    print(f"   - Combined plot: {dataset_name}_training_curves.png")
    print(f"   - Individual plots: {dataset_name}_train_loss.png, {dataset_name}_val_accuracy.png")
    print(f"   - Metrics file: {dataset_name}_training_metrics.txt")
    print(f"   - Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")

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
        total_epoch_loss = 0.9*main_loss + 0.1*aux_loss
        # total_epoch_loss = 0.9*main_loss + 0.099*aux_loss + 5e-3*ot_loss.mean()
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
    save_training_curves(
        train_losses=train_losses,
        val_losses=val_losses, 
        val_accuracies=val_accuracies,
        gamma_values=gamma_discrete_values,
        save_dir=args.save_dir,
        dataset_name=args.dataset.lower()
    )

if __name__ == "__main__":
    main()
