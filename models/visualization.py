import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_token_selection(tokens, selected_mask, step_count=[0]):
    """
    Simple text-based visualization of DPC-KNN token selection effectiveness.
    Only runs once every 100 forward passes to avoid spam.
    """
    step_count[0] += 1
    if step_count[0] % 100 != 1:  # Only visualize on steps 1, 101, 201, etc.
        return
        
    try:
        with torch.no_grad():
            # Work with first batch only
            mask_np = selected_mask[0].cpu().numpy()  # (N,)
            tokens_np = tokens[0].cpu().numpy()  # (N, C)
            
            selected_count = mask_np.sum()
            total_count = len(mask_np)
            selected_indices = np.where(mask_np)[0]
            
            print(f"\n{'='*60}")
            print(f"TOKEN SELECTION ANALYSIS (Step {step_count[0]})")
            print(f"{'='*60}")
            print(f"Selected: {selected_count}/{total_count} tokens ({selected_count/total_count:.1%})")
            print(f"Selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
            
            # Simple diversity analysis
            if selected_count > 1:
                selected_tokens = tokens_np[mask_np]
                # Compute simple statistics
                mean_vals = selected_tokens.mean(axis=1)  # Mean across features for each token
                std_vals = selected_tokens.std(axis=1)   # Std across features for each token
                
                print(f"Selected token statistics:")
                print(f"  Mean feature values: [{mean_vals.min():.3f}, {mean_vals.max():.3f}] (range)")
                print(f"  Std feature values:  [{std_vals.min():.3f}, {std_vals.max():.3f}] (range)")
                
                # Simple pairwise distance
                distances = []
                for i in range(min(5, len(selected_tokens))):  # Only check first 5 to avoid too much computation
                    for j in range(i+1, min(5, len(selected_tokens))):
                        dist = np.sqrt(np.sum((selected_tokens[i] - selected_tokens[j])**2))
                        distances.append(dist)
                
                if distances:
                    print(f"  Avg pairwise distance (first 5): {np.mean(distances):.3f}")
            
            # Distribution check
            indices_spread = selected_indices.max() - selected_indices.min() if len(selected_indices) > 1 else 0
            print(f"Index spread: {indices_spread} (max-min of selected indices)")
            
            # Try to save a simple plot if possible
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-GUI backend
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 4))
                
                # Plot 1: Selection pattern
                plt.subplot(1, 3, 1)
                all_indices = np.arange(total_count)
                colors = ['red' if mask_np[i] else 'lightblue' for i in range(total_count)]
                plt.scatter(all_indices, np.zeros_like(all_indices), c=colors, s=30, alpha=0.7)
                plt.title(f'Selection Pattern (Step {step_count[0]})')
                plt.xlabel('Token Index')
                plt.ylabel('Selected')
                
                # Plot 2: First two feature dimensions
                plt.subplot(1, 3, 2)
                if tokens_np.shape[1] >= 2:
                    plt.scatter(tokens_np[~mask_np, 0], tokens_np[~mask_np, 1], 
                               c='lightblue', alpha=0.6, s=20, label='Not Selected')
                    plt.scatter(tokens_np[mask_np, 0], tokens_np[mask_np, 1], 
                               c='red', s=60, label='Selected', alpha=0.8)
                    plt.xlabel('Feature Dim 0')
                    plt.ylabel('Feature Dim 1')
                    plt.title('Feature Space (2D)')
                    plt.legend()
                
                # Plot 3: Selection stats
                plt.subplot(1, 3, 3)
                plt.bar(['Selected', 'Not Selected'], 
                       [selected_count, total_count - selected_count],
                       color=['red', 'lightblue'], alpha=0.7)
                plt.title('Selection Count')
                plt.ylabel('Number of Tokens')
                
                plt.tight_layout()
                plt.savefig(f'token_selection_step_{step_count[0]}.png', dpi=100, bbox_inches='tight')
                plt.close()
                print(f"✓ Plot saved: token_selection_step_{step_count[0]}.png")
                
            except Exception as plot_error:
                print(f"Plot creation failed (using text only): {plot_error}")
            
            print(f"{'='*60}\n")
            
    except Exception as e:
        print(f"Visualization failed: {e}")

def visualize_I_matrix_range(I_matrix, save_path=None, title="I Matrix Value Distribution"):
    """
    Visualize the range of values in the I matrix with multiple visualization techniques.
    
    Args:
        I_matrix (torch.Tensor): The I matrix to visualize (shape: B x N)
        save_path (str, optional): Path to save the visualization
        title (str): Title for the plots
    """
    # Convert to numpy for plotting
    if isinstance(I_matrix, torch.Tensor):
        I_data = I_matrix.detach().cpu().numpy()
    else:
        I_data = I_matrix
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Flatten the data for overall statistics
    flat_data = I_data.flatten()
    
    # 1. Histogram of all values
    axes[0, 0].hist(flat_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Histogram of I Matrix Values')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Min: {np.min(flat_data):.4f}\nMax: {np.max(flat_data):.4f}\nMean: {np.mean(flat_data):.4f}\nStd: {np.std(flat_data):.4f}'
    axes[0, 0].text(0.7, 0.8, stats_text, transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Box plot for each batch (if multiple batches)
    if I_data.shape[0] > 1:
        axes[0, 1].boxplot([I_data[i] for i in range(min(I_data.shape[0], 10))], 
                          labels=[f'B{i}' for i in range(min(I_data.shape[0], 10))])
        axes[0, 1].set_title('Box Plot by Batch (max 10 batches)')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Value')
    else:
        axes[0, 1].boxplot(flat_data)
        axes[0, 1].set_title('Box Plot of All Values')
        axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Heatmap of the first batch
    im = axes[1, 0].imshow(I_data[0:1], aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Heatmap of First Batch')
    axes[1, 0].set_xlabel('Token Index')
    axes[1, 0].set_ylabel('Batch')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Line plot showing value progression for first few batches
    axes[1, 1].set_title('Value Progression Across Tokens')
    for i in range(min(3, I_data.shape[0])):
        axes[1, 1].plot(I_data[i], label=f'Batch {i}', alpha=0.7)
    axes[1, 1].set_xlabel('Token Index')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== I Matrix Statistics ===")
    print(f"Shape: {I_data.shape}")
    print(f"Min value: {np.min(flat_data):.6f}")
    print(f"Max value: {np.max(flat_data):.6f}")
    print(f"Mean: {np.mean(flat_data):.6f}")
    print(f"Std: {np.std(flat_data):.6f}")
    print(f"Median: {np.median(flat_data):.6f}")
    print(f"25th percentile: {np.percentile(flat_data, 25):.6f}")
    print(f"75th percentile: {np.percentile(flat_data, 75):.6f}")
    
    return fig

def visualize_token_score_distributions(model, dataloader, device, num_samples=1):
    """
    Visualize the distribution of token_scores across the three VTM blocks.
    """
    model.eval()
    
    with torch.no_grad():
        # Get one batch
        batch = next(iter(dataloader))
        tokens = batch[1].to(device).float()
        
        # Forward pass
        _, _, token_score_list = model(tokens, epoch=0, id=0)
        
        # token_score_list contains 3 tensors (one per block)
        # Each has shape [B, N, 1] where B=batch_size, N=num_tokens
        # Flatten across batch and tokens for each block
        block_scores = [scores.cpu().flatten().numpy() for scores in token_score_list]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram for each block
    ax = axes[0, 0]
    for i, scores in enumerate(block_scores):
        ax.hist(scores, bins=50, alpha=0.6, label=f'Block {i+1}')
    ax.set_xlabel('Token Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Token Score Distribution - Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KDE plot
    ax = axes[0, 1]
    for i, scores in enumerate(block_scores):
        sns.kdeplot(scores, ax=ax, label=f'Block {i+1}', linewidth=2)
    ax.set_xlabel('Token Score')
    ax.set_ylabel('Density')
    ax.set_title('Token Score Distribution - KDE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax = axes[1, 0]
    ax.boxplot(block_scores, labels=[f'Block {i+1}' for i in range(3)])
    ax.set_xlabel('VTM Block')
    ax.set_ylabel('Token Score')
    ax.set_title('Token Score Distribution - Box Plot')
    ax.grid(True, alpha=0.3)
    
    # 4. Violin plot
    ax = axes[1, 1]
    parts = ax.violinplot(block_scores, positions=[1, 2, 3], showmeans=True, showmedians=True)
    ax.set_xlabel('VTM Block')
    ax.set_ylabel('Token Score')
    ax.set_title('Token Score Distribution - Violin Plot')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'Block {i+1}' for i in range(3)])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('token_score_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to token_score_distributions.png")
    
    # Print statistics
    print("\n=== Token Score Statistics ===")
    for i, scores in enumerate(block_scores):
        print(f"\nBlock {i+1}:")
        print(f"  Shape: {token_score_list[i].shape}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  25th percentile: {np.percentile(scores, 25):.4f}")
        print(f"  75th percentile: {np.percentile(scores, 75):.4f}")
    
    return block_scores

def visualize_token(x, frame_idx=0, channel_idx=0, batch_idx=0, H=7, W=7, save_path=None, title=None, eigen=None, spectral=False, save_path2=None):
        
        # Extract the specific token
        B, N, C = x.shape
        T = N // (H * W)
        
        # Reshape to [B, T, H, W, C]
        x_reshaped = x.view(B, T, H, W, C)
        
        # Extract specific frame and channel: [H, W]
        feature_map = x_reshaped[batch_idx, frame_idx, :, :, channel_idx].detach().cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(feature_map, cmap='viridis', aspect='auto')
        plt.colorbar(label='Feature Value')
        
        if title is None:
            title = f'Frame {frame_idx}, Channel {channel_idx} (Batch {batch_idx})'
        plt.title(title)
        plt.xlabel(f'Width ({W})')
        plt.ylabel(f'Height ({H})')
        
        # Add text with statistics
        plt.figtext(0.02, 0.02, f'Min: {feature_map.min():.3f}, Max: {feature_map.max():.3f}, Mean: {feature_map.mean():.3f}', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        if eigen is not None:
            eigen = eigen.squeeze(-1)  # Remove last dim if exists
            eigen = eigen.view(B, T, H, W)
            feature_map = eigen[batch_idx, frame_idx, :, :].detach().cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(feature_map, cmap='viridis', aspect='auto')
            plt.colorbar(label='Feature Value')
            
            if title is None:
                title = f'Frame {frame_idx}, Channel {channel_idx} (Batch {batch_idx})'
            plt.title(title)
            plt.xlabel(f'Width ({W})')
            plt.ylabel(f'Height ({H})')
            
            # Add text with statistics
            plt.figtext(0.02, 0.02, f'Min: {feature_map.min():.3f}, Max: {feature_map.max():.3f}, Mean: {feature_map.mean():.3f}', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if save_path2:
                plt.savefig(save_path2, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path2}")
        
        # plt.show()
        
        return feature_map