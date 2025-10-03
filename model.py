import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from attention import MultiHeadAttention
from transformers import SwinModel, ViTModel


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class LearnableVTMBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, partition_factor: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gamma = partition_factor

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        
        self.saliency_head = nn.Linear(dim, 1)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        
    def forward(self, x: torch.Tensor, x_aux: torch.Tensor):
        B, N, C = x.shape
        x_norm = self.norm1(x)

        attn_out, k_for_saliency = self.attn(x_norm)
        x_res = x + attn_out 
        num_targets = N // self.gamma
        if self.training and num_targets > 0 and N > num_targets:
            saliency_scores = torch.tanh(self.saliency_head(k_for_saliency))
            sampling_probs = F.softmax(saliency_scores.squeeze(-1), dim=1)

            target_indices = torch.multinomial(sampling_probs, num_samples=num_targets, replacement=False)
            target_mask = torch.zeros_like(sampling_probs, dtype=torch.bool).scatter_(1, target_indices, True)
            source_mask = ~target_mask
            
            source_tokens = x_res[source_mask].reshape(B, -1, C)
            target_tokens = x_res[target_mask].reshape(B, num_targets, C)
            source_keys = k_for_saliency[source_mask].reshape(B, -1, C)
            target_keys = k_for_saliency[target_mask].reshape(B, num_targets, C)

            similarity = F.cosine_similarity(source_keys.unsqueeze(2), target_keys.unsqueeze(1), dim=-1)
            match_indices = similarity.argmax(dim=2) # (B, num_sources)
            
            merged_tokens = target_tokens.clone()
            match_indices_expanded = match_indices.unsqueeze(-1).expand(-1, -1, C)
            merged_tokens.scatter_add_(1, match_indices_expanded, source_tokens)
            
            counts = torch.ones_like(target_tokens[:, :, 0])
            counts.scatter_add_(1, match_indices, torch.ones_like(match_indices, dtype=torch.float))
            
            x = merged_tokens / counts.unsqueeze(-1)
        else:
            x = x_res

        x = x + self.mlp(self.norm2(x))
        
        aux_loss_contribution = 0
        if self.training:
            x_aux_norm = self.norm1(x_aux)

            _, k_aux = self.attn(x_aux_norm)
            saliency_scores_aux = torch.tanh(self.saliency_head(k_aux))
            
            q_aux, _, v_aux = self.attn.qkv_proj(x_aux_norm).chunk(3, dim=-1)
            
            attn_matrix = torch.bmm(q_aux, k_aux.transpose(1, 2)) / (self.dim ** 0.5)
            attn_matrix = attn_matrix + saliency_scores_aux.transpose(1, 2)
            attn_weights = F.softmax(attn_matrix, dim=-1)
            attn_out_aux = torch.bmm(attn_weights, v_aux)

            x_aux = x_aux + attn_out_aux
            x_aux = x_aux + self.mlp(self.norm2(x_aux))
            
            aux_loss_contribution = saliency_scores_aux.std() * 0.01
        else:
            x_aux = None

        return x, x_aux, aux_loss_contribution


class VideoTokenMergingTransformer(nn.Module):
    def __init__(self, num_classes: int, num_frames: int, patch_dim: int = 1024, num_vtm_blocks: int = 3, num_heads: int = 8, dataset: str = 'LVU'):
        super().__init__()
        self.num_frames = num_frames
        
        if dataset.upper() == 'LVU':
            self.encoder = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
            encoder_dim = 1024
        else:
            self.encoder = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224-in22k')
            encoder_dim = 1024
            
        for param in self.encoder.parameters():
            param.requires_grad = False
                
        self.projection = nn.Linear(encoder_dim, patch_dim)
        
        self.vtm_blocks = nn.ModuleList([
            LearnableVTMBlock(dim=patch_dim, num_heads=num_heads) for _ in range(num_vtm_blocks)
        ])
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, patch_dim))
        
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, num_classes)
        )

    def forward(self, video: torch.Tensor):
        batch_size, num_frames, c, h, w = video.shape
        
        video_flat = video.view(batch_size * num_frames, c, h, w)
        encoder_output = self.encoder(video_flat)
        

        if isinstance(self.encoder, ViTModel):
            frame_features = encoder_output.pooler_output
        else: 
            frame_features = encoder_output.pooler_output
            
        tokens = frame_features.view(batch_size, num_frames, -1)

        tokens = self.projection(tokens)

        tokens += self.pos_embedding[:, :num_frames]
        
        aux_tokens = tokens.clone()
        total_aux_loss = 0.0
        
        for block in self.vtm_blocks:
            tokens, aux_tokens, aux_loss = block(tokens, aux_tokens)
            if self.training:
                total_aux_loss += aux_loss

        final_representation = tokens.mean(dim=1)
        output = self.prediction_head(final_representation)
        
        return output, total_aux_loss