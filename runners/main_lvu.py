import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Add parent directory to path so we can import from models and datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_lvu import VideoTokenMergingTransformer
from datasets.lvu_dataset import CustomDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='LVU Video Token Merging Training')
    parser.add_argument('--dataset', default='lvu', choices=['lvu', 'Breakfast', 'COIN'], type=str, help='Dataset')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of warmup epochs')
    parser.add_argument('--l_secs', default=60, type=int, help='l_secs')
    parser.add_argument('--n_layers', default=3, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=1024, type=int, help='Model dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--d_input', default=1024, type=int, help='Input dimension')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--feature_type', default='vit_spatial', type=str, help='Feature type')
    parser.add_argument('--long_term_task', default='writer', type=str, help='long_term_task')
    parser.add_argument('--num_long_term_classes', default=4, type=int, help='num_long_term_classes')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--save_dir', type=str, default='/data/susimmuk/VideoTokenMerging/LVU/OT', help='Directory to save results')
    return parser.parse_args()

def save_training_curves(train_losses, val_losses, val_accuracies, save_dir, dataset_name='lvu'):
    """
    Save training loss and validation accuracy curves
    """
    save_dir = os.path.join(save_dir, 'training_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
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
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = os.path.join(save_dir, f'{dataset_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
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
        f.write(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}%\n\n")
        
        f.write("Epoch-wise Details:\n")
        f.write("-"*80 + "\n")
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\n")
        f.write("-"*80 + "\n")
        for i, epoch in enumerate(epochs):
            f.write(f"{epoch}\t{train_losses[i]:.6f}\t{val_losses[i]:.6f}\t{val_accuracies[i]:.4f}\n")
    
    print(f"\n📊 Training curves saved to: {save_dir}")
    print(f"   - Combined plot: {dataset_name}_training_curves.png")
    print(f"   - Metrics file: {dataset_name}_training_metrics.txt")
    print(f"   - Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def configure_optimizer(model, lr=0.001, weight_decay=0.01, num_epochs=70, warmup_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def train(args, trainloader, model, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    accum_steps = 1
    optimizer.zero_grad()
    for batch_idx, (video_name_batch, inputs, targets) in pbar:
        inputs, targets = inputs.to(args.device).float(), targets.to(args.device)

        # import time
        # start_time = time.time()
        # optimizer.zero_grad()
        outputs, aux_outputs, ot_loss = model(inputs)

        if args.num_long_term_classes == -1:
            targets = targets.to(torch.float32)
            outputs = outputs[:, 0]
            # aux_output = aux_output[:, 0]

        loss1 = criterion(outputs, targets)
        loss2 = criterion(aux_outputs, targets) 
        # loss = 0.9*loss1 + 0.099 * loss2 
        loss = 0.9*loss1 + 0.099 * loss2 + 5e-3*ot_loss.mean()
        # loss = 0.4*loss1 + 0.1 * loss2 + 0.4*ot_loss.mean()
        # loss = loss1
        # loss.backward()
        # optimizer.step()
        # loss = loss / accum_steps
        loss.backward()
        # end_time = time.time()
        # print(f"Time taken: {end_time - start_time:.2f} seconds")
            # 🔄 update every 4 batches
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # if batch_idx == 2:
            # torch.cuda.synchronize()
            # peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            # print(f"Peak GPU memory: {peak_mem:.2f} GB")
            # exit()
        train_loss += loss.item()
        if args.num_long_term_classes > 0:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if args.num_long_term_classes > 0:
            pbar.set_description(
                '(%d/%d) | Loss: %.3f | Main: %.3f | Aux: %.3f | OT: %.3f | Acc: %.3f%%' %
                (batch_idx, len(trainloader), train_loss / (batch_idx + 1), 0.9*loss1.item(), 0.099*loss2.item(), 1e-3*ot_loss.mean().item(), 100. * correct / total)
            )
            # pbar.set_description(
            #     '(%d/%d) | Loss: %.3f | Main: %.3f | Aux: %.3f | Acc: %.3f%%' %
            #     (batch_idx, len(trainloader), train_loss / (batch_idx + 1), 0.9*loss1.item(), 0.099*loss2.item(), 100. * correct / total)
            # )
        else:
            pbar.set_description(
                '(%d/%d) | Loss: %.3f' %
                (batch_idx, len(trainloader), train_loss / (batch_idx + 1))
            )

def eval(args, dataloader, model, epoch, criterion, split):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    long_term_top1 = 0
    all_preds = []
    long_term_count = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (video_name_batch, inputs, targets) in pbar:
            inputs, targets = inputs.to(args.device).float(), targets.to(args.device)   #.contiguous()
            outputs = model(inputs)

            if args.num_long_term_classes == -1:
                targets = targets.to(torch.float32)
                outputs = outputs[:, 0]

            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            if args.num_long_term_classes > 0:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            lt_pred = outputs.cpu()
            lt_labels = targets

            all_preds.append((video_name_batch, lt_pred, lt_labels))

            if args.num_long_term_classes > 0:
                long_term_top1 += correct
                long_term_count += targets.shape[0]

            if args.num_long_term_classes > 0:
                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (batch_idx, len(dataloader), eval_loss / (batch_idx + 1), 100. * correct / total, correct, total)
                )
            else:
                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f' %
                    (batch_idx, len(dataloader), eval_loss / (batch_idx + 1))
                )

    clip_mse = []
    split_result = {}
    pred_agg = {}
    video_label = {}

    for video_name_batch, pred_batch, label_batch in all_preds:
        for i in range(len(video_name_batch)):
            v_name = video_name_batch[i]
            if v_name not in pred_agg:
                if args.num_long_term_classes > 0:
                    pred_agg[v_name] = softmax(pred_batch[i])
                else:
                    pred_agg[v_name] = [pred_batch[i]]
                video_label[v_name] = label_batch[i].cpu()
            else:
                if args.num_long_term_classes > 0:
                    pred_agg[v_name] += softmax(pred_batch[i])
                else:
                    pred_agg[v_name].append(pred_batch[i])

                assert video_label[v_name] == label_batch[i].cpu()

            if args.num_long_term_classes == -1:
                clip_mse.append(
                    (pred_batch[i] - label_batch[i]) ** 2.0
                )

    agg_sm_correct, agg_count = 0.0, 0.0
    mse = []

    for v_name in pred_agg.keys():
        if args.num_long_term_classes > 0:
            if pred_agg[v_name].argmax() == video_label[v_name]:
                agg_sm_correct += 1
        else:
            mse.append(
                (np.mean(pred_agg[v_name]) - video_label[v_name]) ** 2.0
            )
        agg_count += 1
        if args.num_long_term_classes > 0:
            acc = 100.0 * agg_sm_correct / agg_count
            split_result[split] = f'{acc} {agg_sm_correct} {agg_count}'
        else:
            split_result[split] = f'{np.mean(mse)} {len(mse)}'

    print(split_result)

    with open(args.output_eval_file, "a") as writer:
        if split == 'val':
            writer.write("Epoch trained %s\n" % str(epoch))
        for key in sorted(split_result.keys()):
            writer.write("%s = %s\n" % (key, str(split_result[key])))
    writer.close()

    if args.num_long_term_classes > 0:
        return acc
    else:
        return np.mean(mse)

def main():
    tasks = [('relationship', 4), ('way_speaking', 5), ('scene', 6), ('director', 10),
            ('writer', 10), ('year', 9), ('like_ratio', -1), ('view_count', -1), ('genre', 4)]

    # tasks = [('director', 10)]
    set_seed(1112)
    print('Device', torch.cuda.device_count())
    
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.num_long_term_classes = tasks[3][1]
    args.long_term_task = tasks[3][0]
    
    # Set up output directories
    if args.num_long_term_classes > 0:
        args.d_output = args.num_long_term_classes
    else:
        args.d_output = 1

    # if args.feature_type == 'vit_spatial':
    #     args.l_max = args.l_secs*197

    args.out_dir = os.path.join(args.save_dir, 'outputs')
    os.makedirs(args.out_dir, exist_ok=True)
    args.output_eval_file = os.path.join(args.out_dir, f'{args.long_term_task}.txt')

    print(args)
    print(f'==> Preparing {args.dataset} data..')

    # Create datasets
    trainset = CustomDataset(args=args, split='train')
    valset = CustomDataset(args=args, split='val')
    testset = CustomDataset(args=args, split='test')

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.dataset.upper() == 'LVU':
        num_frames = args.l_secs
        num_tokens = 16 * 16 * num_frames  # ViT-L: 16x16 patches
    else:
        num_frames = 64
        num_tokens = 7 * 7 * num_frames   
    patch_dim = 1024
    # Model
    print('==> Building model..')
    # model = HybridVideoTokenMergingTransformer(
    #     num_classes=args.num_long_term_classes,
    #     num_tokens=num_tokens,
    #     patch_dim=patch_dim,
    #     num_vtm_blocks=3,
    #     num_heads=8
    # )
    model = VideoTokenMergingTransformer(
        num_classes=args.num_long_term_classes,
        num_tokens=num_tokens,
        patch_dim=patch_dim,
        num_vtm_blocks=3,
        num_heads=8
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    model = model.to(args.device)

    # def gpu_model_memory_bytes(model):
    #     return sum(
    #         p.numel() * p.element_size()
    #         for p in model.parameters()
    #         if p.is_cuda
    #     )
    # mem_bytes = gpu_model_memory_bytes(model)
    # mem_mb = mem_bytes / 1024**2
    # print(f"Model parameter memory: {mem_mb:.2f} MB")
    # exit()

    if args.num_long_term_classes > 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer, scheduler = configure_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, num_epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )

    start_epoch = 0
    best_val_acc = 0
    best_test_acc = 0  

    if args.evaluate:
        print('==> Evaluating model..')
        checkpoint = torch.load(f'/home/csgrad/susimmuk/long-video/best_model_breakfast_dpc_knn.pth', map_location=args.device)
        model.load_state_dict(checkpoint)
        print('Model loaded.')

        val_acc = eval(args=args, dataloader=valloader, model=model,
                       epoch=0, criterion=criterion, split='val')
        test_acc = eval(args=args, dataloader=testloader, model=model,
                   epoch=0, criterion=criterion, split='test')
        return
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    pbar = tqdm(range(start_epoch, start_epoch + args.epochs))
    for epoch in pbar:
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        pbar.set_description('Epoch: %d' % (epoch))

        # Training
        train(args=args, trainloader=trainloader, model=model, optimizer=optimizer, criterion=criterion)
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        for i, param_group in enumerate(optimizer.param_groups):
            print(f'learning rate param group {i}', param_group['lr'])
            
        # Evaluate every 10 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print('Result for epoch ', epoch + 1)
            val_acc = eval(args=args, dataloader=valloader, model=model,
                           epoch=epoch + 1, criterion=criterion, split='val')
            test_acc = eval(args=args, dataloader=testloader, model=model,
                       epoch=epoch + 1, criterion=criterion, split='test')
            
            # Store metrics for plotting
            val_accuracies.append(val_acc if args.num_long_term_classes > 0 else -val_acc)  # For MSE, we want lower values
            
            with open(args.output_eval_file, "a") as writer:
                for i, param_group in enumerate(optimizer.param_groups):
                    lr = param_group['lr']
                    print(f'learning rate param group {i} : {lr}')
                    writer.write(f'learning rate param group {i} : {lr}')
                writer.write('\n\n')
            writer.close()
                
            # Save best model
            if args.num_long_term_classes > 0:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(args.save_dir, f'best_model_{args.long_term_task}_lvu_{best_test_acc:.4f}.pth'))
            else:
                if val_acc < best_val_acc or best_val_acc == 0:  # For MSE, lower is better
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(args.save_dir, f'best_model_{args.long_term_task}_lvu_{best_test_acc:.4f}.pth'))

    print(f"\n✅ Training complete. Best validation result: {best_val_acc:.4f}")
    print(f"✅ Best test result: {best_test_acc:.4f}")

    # Save training curves if we have enough data
    # if len(val_accuracies) > 0:
    #     # Create dummy train/val losses for plotting (since we don't track them every epoch)
    #     train_losses = [0.5 - 0.01*i for i in range(len(val_accuracies))]  # Dummy decreasing loss
    #     val_losses = [0.4 - 0.008*i for i in range(len(val_accuracies))]   # Dummy decreasing loss
        
    #     save_training_curves(
    #         train_losses=train_losses,
    #         val_losses=val_losses, 
    #         val_accuracies=val_accuracies,
    #         save_dir='./training_plots',
    #         dataset_name=f'{args.dataset}_{args.long_term_task}'
    #     )

if __name__ == "__main__":
    main()