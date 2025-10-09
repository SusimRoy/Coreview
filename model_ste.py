import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
from attention import MultiHeadAttention
from transformers import SwinModel, ViTModel


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        # self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.fc2(x)
        return x

class LearnableVTMBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_heads: int, partition_factor: int = 6, 
                 mlp_ratio: float = 4.0, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = partition_factor
        self.temperature = nn.Parameter(torch.tensor(temperature))  # Learnable temperature

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        self.saliency_head = nn.Linear(dim, 1, bias=False)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
        self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
    
    def gumbel_top_k_selection(self, saliency_logits, k):
        """Gumbel-Softmax top-k selection"""
        B, N = saliency_logits.shape
        
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(saliency_logits) + 1e-8) + 1e-8)
        gumbel_logits = (saliency_logits + gumbel_noise) / self.temperature
        
        # Get top-k indices  
        _, top_k_indices = torch.topk(gumbel_logits, k, dim=-1)
        
        # Create hard mask with straight-through gradients
        hard_mask = torch.zeros_like(saliency_logits).scatter_(-1, top_k_indices, 1.0)
        soft_weights = F.softmax(gumbel_logits, dim=-1)
        
        # Straight-through estimator
        target_mask = hard_mask - soft_weights.detach() + soft_weights
        
        return target_mask.bool(), top_k_indices
        
    def forward(self, x: torch.Tensor, x_aux: torch.Tensor):
        # === Main Path ===
        x_norm = self.norm1(x)
        x_prime, K_matrix = self.attn1(x_norm)
        x_prime = nn.Dropout(0.1)(x_prime)
        x_res = x + x_prime
        
        # Use raw logits instead of tanh for Gumbel-Softmax
        saliency_logits = self.saliency_head(K_matrix).squeeze(-1)  # (B, N)
        
        B, N, C = x.shape
        num_targets = N // self.gamma
        
        # Gumbel-Softmax selection
        target_mask, target_indices = self.gumbel_top_k_selection(saliency_logits, num_targets)
        source_mask = ~target_mask
        
        # Rest of your code remains the same...
        source_tokens = x_res[source_mask].reshape(B, -1, C)
        target_tokens = x_res[target_mask].reshape(B, num_targets, C)
        source_keys = K_matrix[source_mask].reshape(B, -1, C)
        target_keys = K_matrix[target_mask].reshape(B, num_targets, C)
        
        similarity = F.cosine_similarity(source_keys.unsqueeze(2), target_keys.unsqueeze(1), dim=-1)
        match_indices = similarity.argmax(dim=2)
        
        merged_tokens = target_tokens.clone()
        match_indices_expanded = match_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_tokens.scatter_add_(1, match_indices_expanded, source_tokens)
        
        counts = torch.ones_like(target_tokens[:, :, 0])
        counts.scatter_add_(1, match_indices, torch.ones_like(match_indices, dtype=torch.float))
        merged_main = merged_tokens / counts.unsqueeze(-1)
        merged = self.mlp1(merged_main)
        
        # Auxiliary path handling...
        if self.training:
            # Similar modifications for auxiliary path
            x_aux_norm = self.norm2(x_aux)
            q_aux, k_aux, v_aux = self.qkv_proj(x_aux_norm).chunk(3, dim=-1)
            attn_matrix = torch.bmm(q_aux, k_aux.transpose(1, 2)) / (self.dim ** 0.5)
            attn_matrix = attn_matrix + saliency_logits.unsqueeze(1)  # Use logits not tanh
            attn_weights = F.softmax(attn_matrix, dim=-1)
            attn_out_aux = torch.bmm(attn_weights, v_aux)
            x_aux = x_aux + attn_out_aux
            
            source_tokens_aux = x_aux[source_mask].reshape(B, -1, C)
            target_tokens_aux = x_aux[target_mask].reshape(B, num_targets, C)
            merged_tokens_aux = target_tokens_aux.clone()
            merged_tokens_aux.scatter_add_(1, match_indices_expanded, source_tokens_aux)
            merged_aux = merged_tokens_aux / counts.unsqueeze(-1)
            merged_aux = self.mlp2(merged_aux)
            return merged, merged_aux
        else:
            return merged, None


class VideoTokenMergingTransformer(nn.Module):
    def __init__(self, num_classes: int, num_tokens: int, patch_dim: int = 1024, num_vtm_blocks: int = 3, num_heads: int = 8):
        super().__init__()

        dims = [patch_dim // (2**i) for i in range(num_vtm_blocks + 1)]  # [1024, 512, 256, 128]
        self.vtm_blocks = nn.ModuleList([
            LearnableVTMBlock(dim=dims[i], out_dim=dims[i+1], num_heads=num_heads) for i in range(num_vtm_blocks)
        ])
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, patch_dim))
        final_dim = dims[num_vtm_blocks]  # Final dimension after all VTM blocks
        # self.prediction_head = nn.Sequential(
        #     nn.Linear(final_dim, num_classes)
        # )
        self.prediction_head1 = nn.Sequential(
            nn.Linear(final_dim, num_classes)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(final_dim, num_classes)
        )

    def forward(self, tokens: torch.Tensor):
        # tokens: (batch_size, num_tokens, patch_dim)
        # tokens = tokens + self.pos_embedding[:, :tokens.size(1)]
        aux_tokens = tokens.clone()
        for block in self.vtm_blocks:
            tokens, aux_tokens = block(tokens, aux_tokens)
        if self.training:
            final_representation1 = tokens.mean(dim=1)
            output = self.prediction_head1(final_representation1)
            final_representation2 = aux_tokens.mean(dim=1)
            aux_output = self.prediction_head2(final_representation2)
            return output, aux_output  
        else:
            final_representation = tokens.mean(dim=1)
            output = self.prediction_head1(final_representation)
            return output