import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.fc1(x))

class LearnableVTMBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_heads: int, partition_factor: int = 6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.gamma = partition_factor
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        self.saliency_head = nn.Linear(dim, 1, bias=False)
        self.mlp1 = MLP(dim, int(dim * 4.0), out_dim)

    def forward(self, x: torch.Tensor, x_aux: torch.Tensor = None, return_map: bool = False):
        # === Main Path ===
        x_norm = self.norm1(x)
        x_prime, K_matrix = self.attn1(x_norm)
        x_res = x + x_prime
        
        # Saliency & Sampling
        saliency_scores = torch.tanh(self.saliency_head(K_matrix))
        B, N, C = x.shape
        num_targets = N // self.gamma
        
        sampling_probs = F.softmax(saliency_scores.squeeze(-1), dim=1)
        target_indices = torch.multinomial(sampling_probs, num_samples=num_targets, replacement=False)
        
        # Create masks
        target_mask = torch.zeros_like(sampling_probs, dtype=torch.bool).scatter_(1, target_indices, True)
        source_mask = ~target_mask
        
        # Split
        source_tokens = x_res[source_mask].reshape(B, -1, C)
        target_tokens = x_res[target_mask].reshape(B, num_targets, C)
        source_keys = K_matrix[source_mask].reshape(B, -1, C)
        target_keys = K_matrix[target_mask].reshape(B, num_targets, C)
        
        # Matching
        source_keys_norm = F.normalize(source_keys, p=2, dim=-1)
        target_keys_norm = F.normalize(target_keys, p=2, dim=-1)
        similarity = torch.bmm(source_keys_norm, target_keys_norm.transpose(1, 2)) 
        match_indices = similarity.argmax(dim=2) # (B, num_sources) -> index in 0..num_targets

        # --- Map Tracking for Visualization ---
        token_map = None
        if return_map:
            # We construct a map (B, N) where value is the target index in the next layer
            token_map = torch.zeros((B, N), dtype=torch.long, device=x.device)
            
            # 1. Targets map to themselves (re-indexed to 0..num_targets)
            # We need to broadcast the batch index correctly
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_targets)
            token_map[batch_indices, target_indices] = torch.arange(num_targets, device=x.device).expand(B, -1)
            
            # 2. Sources map to their matched targets
            # Iterate batch to safely scatter source indices (since source count is constant per batch but indices vary)
            flat_source_mask = source_mask.view(-1)
            for b in range(B):
                src_idxs = torch.nonzero(source_mask[b], as_tuple=True)[0]
                token_map[b, src_idxs] = match_indices[b]
        # --------------------------------------

        # Merging
        merged_tokens = target_tokens.clone()
        match_indices_expanded = match_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_tokens.scatter_add_(1, match_indices_expanded, source_tokens)
        
        counts = torch.ones_like(target_tokens[:, :, 0])
        counts.scatter_add_(1, match_indices, torch.ones_like(match_indices, dtype=torch.float))
        
        merged_main = merged_tokens / counts.unsqueeze(-1)
        merged = self.mlp1(merged_main)
        
        return merged, x_aux, token_map

class VideoTokenMergingTransformer(nn.Module):
    def __init__(self, num_classes: int, num_tokens: int, patch_dim: int = 1024, num_vtm_blocks: int = 3, num_heads: int = 8):
        super().__init__()
        dims = [patch_dim // (2**i) for i in range(num_vtm_blocks + 1)]
        self.vtm_blocks = nn.ModuleList([
            LearnableVTMBlock(dim=dims[i], out_dim=dims[i+1], num_heads=num_heads) for i in range(num_vtm_blocks)
        ])
        self.prediction_head1 = nn.Sequential(nn.Linear(dims[num_vtm_blocks], num_classes))

    def forward(self, tokens: torch.Tensor, return_maps=False):
        aux_tokens = tokens.clone()
        maps = []
        
        for block in self.vtm_blocks:
            tokens, aux_tokens, blk_map = block(tokens, aux_tokens, return_map=return_maps)
            if return_maps:
                maps.append(blk_map)

        output = self.prediction_head1(tokens.mean(dim=1))
        
        if return_maps:
            return output, maps
        return output