import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
# from models.dpc_knn import DPCKNNTokenMerger
from models.dpc_knn import DPCKNNTokenSelector
import matplotlib.pyplot as plt
import numpy as np
from models.visualization import visualize_token_selection

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
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class HybridVTMBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_heads: int, 
                 k_neighbors: int = 7, target_ratio: float = 0.3, mlp_ratio: float = 4.0, drop_path_rate: float = 0.1, num_classes: int = 51, gamma: int = 10):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.target_ratio = target_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        
        # DPC-KNN selector for token selection
        self.token_selector = DPCKNNTokenSelector(k_neighbors=k_neighbors)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        # self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        # self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
        # self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
        self.alpha = 0.5
        self.gamma = gamma
    
    def info_maximize(self, I, K, iters, p):
        X = torch.ones_like(I) / I.size(1)
        for _ in range(iters):
            # Use batch matrix-vector multiplication instead of torch.dot
            KX = torch.bmm(K, X.unsqueeze(-1)).squeeze(-1)
            X = torch.softmax(p*I - 2*p*self.alpha*KX, dim=-1)
        return X
        
    def forward(self, x: torch.Tensor, epoch: int, id: int):
        # === Main Path ===
        # Main Path Normalization
        x_norm = self.norm1(x)
        x_prime, K_matrix = self.attn1(x_norm)
        x_res = x + self.drop_path(x_prime)
        B, N, C = x.shape

        normalized_tokens = F.normalize(K_matrix, p=2, dim=-1)
        K = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))
        I_matrix, closest_center_idx = self.token_selector.get_token_scores(K_matrix, self.target_ratio)

        # Merge the tokens to the selected ones based on the cluster they belong to. Don't merge all the tokens.
        # Prune the tokens which are far away from any cluter center.
            
        p = N // self.gamma
        X_matrix = self.info_maximize(I_matrix, K, iters=20, p=p)
        _, top_indices = torch.topk(X_matrix, k=p, dim=-1)
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        selected_mask.scatter_(1, top_indices, True)
        source_mask = ~selected_mask
        num_targets = selected_mask.sum(dim=1)[0].item()  # Assuming same count per batch
        
        # Get cluster assignments for all tokens
        source_cluster_idx = closest_center_idx[source_mask].reshape(B, -1)  # (B, num_sources)
        target_cluster_idx = closest_center_idx[selected_mask].reshape(B, num_targets)  # (B, num_targets)
        
        source_tokens = x_res[source_mask].reshape(B, -1, C)
        target_tokens = x_res[selected_mask].reshape(B, num_targets, C)
        
        # Vectorized cluster-based merging
        merged_tokens = target_tokens.clone()
        counts = torch.ones_like(target_tokens[:, :, 0])
        
        # Create comparison matrix: (B, num_sources, num_targets)
        # cluster_match[b, s, t] = True if source s and target t belong to same cluster
        cluster_match = source_cluster_idx.unsqueeze(2) == target_cluster_idx.unsqueeze(1)  # (B, num_sources, num_targets)
        
        # For each source token, find the first target token in the same cluster
        # Use argmax to get the first True occurrence (first matching target)
        match_indices = cluster_match.float().argmax(dim=2)  # (B, num_sources)
        
        # Only keep matches where there actually is a match (cluster_match.any(dim=2))
        valid_matches = cluster_match.any(dim=2)  # (B, num_sources)
        
        # Expand match_indices for scatter_add operation
        match_indices_expanded = match_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, num_sources, C)
        
        # Only add source tokens that have valid cluster matches
        valid_source_tokens = source_tokens * valid_matches.unsqueeze(-1)  # (B, num_sources, C)
        
        # Scatter add the source tokens to their matching target positions
        merged_tokens.scatter_add_(1, match_indices_expanded, valid_source_tokens)
        
        # Update counts for averaging
        valid_match_counts = valid_matches.float()  # (B, num_sources)
        counts.scatter_add_(1, match_indices, valid_match_counts)
        
        merged_main = merged_tokens / counts.unsqueeze(-1)
        merged = self.mlp1(merged_main)

        # normalized_tokens = F.normalize(K_matrix, p=2, dim=-1)
        # K = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))
        # I_matrix = self.token_selector.get_token_scores(K_matrix, self.target_ratio)

        # p = N // self.gamma
        # X_matrix = self.info_maximize(I_matrix, K, iters=20, p=p)
        # # if (self.training) and epoch % 10 == 0 and id == 50:
        # #     visualize_I_matrix_range(I_matrix, save_path="I_matrix_visualization.png", 
        # #                             title="I Matrix Value Distribution")
        # #     visualize_I_matrix_range(X_matrix, save_path="X_matrix_visualization.png", 
        # #                             title="X Matrix Value Distribution")
        # _, top_indices = torch.topk(X_matrix, k=p, dim=-1)
        # selected_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        # selected_mask.scatter_(1, top_indices, True)
        
        # # Simple pruning: just keep the selected tokens, discard the rest
        # num_targets = selected_mask.sum(dim=1)[0].item()  # Assuming same count per batch
        # target_tokens = x_res[selected_mask].reshape(B, num_targets, C)
        
        # # Apply MLP to the pruned tokens
        # pruned = self.mlp1(target_tokens)
        
        # return pruned

        
        return merged


class HybridVideoTokenMergingTransformer(nn.Module):
    def __init__(self, num_classes: int, num_tokens: int, patch_dim: int = 1024, 
                 num_vtm_blocks: int = 2, num_heads: int = 8, k_neighbors: int = 7, 
                 target_ratio: float = 0.3):
        super().__init__()

        dims = [patch_dim // (2**i) for i in range(num_vtm_blocks + 1)]  # [1024, 512, 256, 128]
        
        # Progressive reduction ratios
        target_ratios = [target_ratio *(0.5**i) for i in range(num_vtm_blocks)]
        target_ratios = [max(0.1, ratio) for ratio in target_ratios]
        k_neighbors = [8,5,3]
        drop_path_rates = [0.1, 0.1, 0.1]
        gamma = [10,10,6]
        
        self.vtm_blocks = nn.ModuleList([
            HybridVTMBlock(
                dim=dims[i], 
                out_dim=dims[i+1], 
                num_heads=num_heads,
                k_neighbors=k_neighbors[i],
                target_ratio=target_ratios[i],
                drop_path_rate=drop_path_rates[i],
                gamma=gamma[i]
            ) for i in range(num_vtm_blocks)
        ])
        
        final_dim = dims[num_vtm_blocks]
        self.prediction_head1 = nn.Sequential(
            nn.Linear(final_dim, num_classes)
        )
        # self.prediction_head2 = nn.Sequential(
        #     nn.Linear(final_dim, num_classes)
        # )
    
    # init linear layers weigths with kaiming normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        # self.score_vector = nn.Parameter(torch.randn(num_classes, patch_dim))

    def ema(self, v1, v2, alpha=0.9):
        return alpha * v1 + (1 - alpha) * v2

    def forward(self, tokens: torch.Tensor, epoch: int, id: int):
        for block in self.vtm_blocks:
            tokens = block(tokens, epoch, id)
        
        final_representation = tokens.mean(dim=1)
        # final_representation = self.ema(final_representation, self.score_vector, alpha=0.9)
        output = self.prediction_head1(final_representation)
        return output