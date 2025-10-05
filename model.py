import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
from attention import MultiHeadAttention
from transformers import SwinModel, ViTModel


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc1(x))

class LearnableVTMBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, partition_factor: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gamma = partition_factor

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        self.saliency_head = nn.Linear(dim, 1)
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        
    def forward(self, x: torch.Tensor, x_aux: torch.Tensor):
        # === Main Path ===
        # Main Path Normalization
        x_norm = self.norm1(x)
        # Attention and K matrix extraction
        x_prime, K_matrix = self.attn1(x_norm)
        x_res = x + x_prime
        saliency_scores = torch.tanh(self.saliency_head(K_matrix))  # (B, N, 1)
        B, N, C = x.shape
        num_targets = N // self.gamma
        sampling_probs = F.softmax(saliency_scores.squeeze(-1), dim=1)  # (B, N)
        target_indices = torch.multinomial(sampling_probs, num_samples=num_targets, replacement=False)
        target_mask = torch.zeros_like(sampling_probs, dtype=torch.bool).scatter_(1, target_indices, True)
        source_mask = ~target_mask
        source_tokens = x_res[source_mask].reshape(B, -1, C)
        target_tokens = x_res[target_mask].reshape(B, num_targets, C)
        source_keys = K_matrix[source_mask].reshape(B, -1, C)
        target_keys = K_matrix[target_mask].reshape(B, num_targets, C)
        similarity = F.cosine_similarity(source_keys.unsqueeze(2), target_keys.unsqueeze(1), dim=-1)
        match_indices = similarity.argmax(dim=2) # (B, num_sources)
        merged_tokens = target_tokens.clone()
        match_indices_expanded = match_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_tokens.scatter_add_(1, match_indices_expanded, source_tokens)
        counts = torch.ones_like(target_tokens[:, :, 0])
        counts.scatter_add_(1, match_indices, torch.ones_like(match_indices, dtype=torch.float))
        merged_main = merged_tokens / counts.unsqueeze(-1)
        merged = self.mlp1(merged_main)
        # === Auxiliary Path ===
        if self.training:
            x_aux_norm = self.norm2(x_aux)
            q_aux, k_aux, v_aux = self.qkv_proj(x_aux_norm).chunk(3, dim=-1)
            attn_matrix = torch.bmm(q_aux, k_aux.transpose(1, 2)) / (self.dim ** 0.5)
            attn_matrix = attn_matrix + saliency_scores.transpose(1, 2)
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
        # self.num_frames = num_frames
        
        # if dataset.upper() == 'LVU':
        #     self.encoder = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        #     encoder_dim = 1024
        # else:
        #     self.encoder = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224-in22k')
        #     encoder_dim = 1024
            
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
                        
        self.vtm_blocks = nn.ModuleList([
            LearnableVTMBlock(dim=patch_dim, num_heads=num_heads) for _ in range(num_vtm_blocks)
        ])
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, patch_dim))
        self.prediction_head = nn.Sequential(
            nn.Linear(patch_dim, num_classes)
        )

    def forward(self, tokens: torch.Tensor):
        # tokens: (batch_size, num_tokens, patch_dim)
        tokens = tokens + self.pos_embedding[:, :tokens.size(1)]
        aux_tokens = tokens.clone()
        for block in self.vtm_blocks:
            tokens, aux_tokens = block(tokens, aux_tokens)
        final_representation = tokens.mean(dim=1)
        output = self.prediction_head(final_representation)
        return output