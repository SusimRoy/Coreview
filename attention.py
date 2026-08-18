# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from models.utils import trunc_normal_
# from entmax import sparsemax, entmax15, entmax_bisect, normmax_bisect, budget_bisect
# class MultiHeadAttention(nn.Module):
#     """
#     A custom implementation of Multi-Head Attention.
#     This module allows us to directly access the Key (K) vectors for saliency estimation,
#     which is not straightforward with the standard nn.MultiheadAttention module.
#     """
#     def __init__(self, dim: int, num_heads: int, batch_first: bool = True):
#         super().__init__()
#         assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.batch_first = batch_first

#         # Linear projections for Q, K, V from a single input
#         # self.qkv_proj = nn.Linear(dim, dim * 3)
#         self.q = nn.Linear(dim, dim, bias=False)
#         self.kv = nn.Linear(dim, dim * 2, bias=False)
#         # Output projection
#         self.out_proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(0.1)
#         self.scale = self.head_dim ** -0.5
#         self.topk = 50

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x: torch.Tensor, token_dict=None, out_dict=None):
#         if self.batch_first:
#             batch_size, seq_length, _ = x.shape
#         else:
#             seq_length, batch_size, _ = x.shape

#         # Method where first cluster then attention
#         Nq = x.shape[1]
#         x_down = out_dict['x']
#         x_og = token_dict['x']
#         q = self.q(x_down).reshape(batch_size, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
#         kv = self.kv(x_og).reshape(batch_size, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
#         k, v = kv[0], kv[1]
#         attn = (q * self.scale) @ k.transpose(-2, -1)
#         token_score = token_dict['token_score'].squeeze(-1)[:, None, None, :]
#         attn = attn+token_score
#         attn = normmax_bisect(attn, alpha=2, dim=-1)
#         # attn = attn.softmax(dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(batch_size, Nq, self.dim)
#         x = self.out_proj(x)
#         x = self.proj_drop(x)

#         # Method where first attention then cluster

        

#         return x, k


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
        attn_output = self.out_proj(attn_output)

        # Return the final output and the original key vectors (before reshaping)
        return attn_output, k
