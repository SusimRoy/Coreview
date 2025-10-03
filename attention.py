import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    A custom implementation of Multi-Head Attention.
    This module allows us to directly access the Key (K) vectors for saliency estimation,
    which is not straightforward with the standard nn.MultiheadAttention module.
    """
    def __init__(self, dim: int, num_heads: int, batch_first: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.batch_first = batch_first

        # Linear projections for Q, K, V from a single input
        self.qkv_proj = nn.Linear(dim, dim * 3)
        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for Multi-Head Attention.
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, N, C) if batch_first, else (N, B, C).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The output of the attention mechanism.
                - The Key (K) tensor, for use in saliency estimation.
        """
        if self.batch_first:
            batch_size, seq_length, _ = x.shape
        else:
            seq_length, batch_size, _ = x.shape

        # 1. Project to Q, K, V
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # 2. Reshape for multi-head computation
        # (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k_reshaped = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k_reshaped, v)

        # 4. Reshape and project output
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dim)
        output = self.out_proj(attn_output)

        # Return the final output and the original key vectors (before reshaping)
        return output, k
