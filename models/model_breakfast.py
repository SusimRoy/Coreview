import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

# --- Standard Block with Tracking ---
class LearnableVTMBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_heads: int, partition_factor: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.gamma = partition_factor
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        self.saliency_head = nn.Linear(dim, 1, bias=False)
        self.mlp1 = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), out_dim))

    def forward(self, x, x_aux=None, return_map=False):
        B, N, C = x.shape
        x_norm = self.norm1(x)
        _, K_matrix = self.attn1(x_norm)
        
        # Saliency
        saliency_scores = torch.tanh(self.saliency_head(K_matrix))
        sampling_probs = F.softmax(saliency_scores.squeeze(-1), dim=1)
        num_targets = N // self.gamma
        
        # Selection
        target_indices = torch.multinomial(sampling_probs, num_samples=num_targets, replacement=False)
        target_mask = torch.zeros((B, N), dtype=torch.bool, device=x.device).scatter_(1, target_indices, True)
        source_mask = ~target_mask
        
        # Matching
        source_keys = K_matrix[source_mask].reshape(B, -1, C)
        target_keys = K_matrix[target_mask].reshape(B, num_targets, C)
        
        source_keys_norm = F.normalize(source_keys, p=2, dim=-1)
        target_keys_norm = F.normalize(target_keys, p=2, dim=-1)
        
        # similarity: (B, num_sources, num_targets)
        similarity = torch.bmm(source_keys_norm, target_keys_norm.transpose(1, 2))
        match_indices = similarity.argmax(dim=2) 
        
        # --- TRACKING ---
        token_map = None
        if return_map:
            # map[b, old_idx] = new_idx
            token_map = torch.zeros((B, N), dtype=torch.long, device=x.device)
            # Targets map to 0..num_targets
            new_tgt_idxs = torch.arange(num_targets, device=x.device).expand(B, num_targets)
            token_map.scatter_(1, target_indices, new_tgt_idxs)
            # Sources map to their match
            for b in range(B):
                src_locs = torch.nonzero(source_mask[b]).squeeze(-1)
                token_map[b, src_locs] = match_indices[b]
        # ----------------
        
        # Merging
        source_tokens = x[source_mask].reshape(B, -1, C)
        target_tokens = x[target_mask].reshape(B, num_targets, C)
        
        merged = target_tokens.clone()
        merged.scatter_add_(1, match_indices.unsqueeze(-1).expand(-1, -1, C), source_tokens)
        
        # Count averaging
        counts = torch.ones((B, num_targets, 1), device=x.device)
        counts.scatter_add_(1, match_indices.unsqueeze(-1), torch.ones_like(match_indices.unsqueeze(-1), dtype=torch.float))
        
        merged = self.mlp1(merged / counts)
        return merged, x_aux, token_map

class VideoTokenMergingTransformer(nn.Module):
    def __init__(self, num_classes=10, num_tokens=3136, patch_dim=1024, num_vtm_blocks=3):
        super().__init__()
        dims = [patch_dim // (2**i) for i in range(num_vtm_blocks + 1)]
        self.vtm_blocks = nn.ModuleList([
            LearnableVTMBlock(dim=dims[i], out_dim=dims[i+1], num_heads=4) for i in range(num_vtm_blocks)
        ])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x, return_maps=False):
        maps = []
        aux = x.clone()
        for blk in self.vtm_blocks:
            x, aux, m = blk(x, aux, return_map=return_maps)
            if return_maps: maps.append(m)
        
        if return_maps: return self.head(x.mean(1)), maps
        return self.head(x.mean(1))