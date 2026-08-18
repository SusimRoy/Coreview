import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
from attention import MultiHeadAttention
# from transformers import SwinModel, ViTModel
import ot


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
    def __init__(self, dim: int, out_dim: int, num_heads: int, partition_factor: int = 10, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = partition_factor

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads=num_heads, batch_first=True)
        self.saliency_head = nn.Linear(dim, 1, bias=False)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
        self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
        
    def forward(self, x: torch.Tensor, x_aux: torch.Tensor):
        # === Main Path ===
        # Main Path Normalization
        x_norm = self.norm1(x)
        # Attention and K matrix extraction
        x_prime, K_matrix = self.attn1(x_norm)
        x_prime = nn.Dropout(0.1)(x_prime)
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
        source_keys_norm = F.normalize(source_keys, p=2, dim=-1)  # (B, num_sources, C)
        target_keys_norm = F.normalize(target_keys, p=2, dim=-1)  # (B, num_targets, C)
        similarity = torch.bmm(source_keys_norm, target_keys_norm.transpose(1, 2)) 
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

        dims = [patch_dim // (2**i) for i in range(num_vtm_blocks + 1)]  # [1024, 512, 256, 128]
        self.vtm_blocks = nn.ModuleList([
            LearnableVTMBlock(dim=dims[i], out_dim=dims[i+1], num_heads=num_heads) for i in range(num_vtm_blocks)
        ])
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, patch_dim))
        final_dim = dims[num_vtm_blocks]  # Final dimension after all VTM blocks
        self.prediction_head1 = nn.Sequential(
            nn.Linear(final_dim, num_classes)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(final_dim, num_classes)
        )
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], patch_dim//2),
                nn.GELU(),
                nn.Linear(patch_dim//2, patch_dim//2)
            ) for i in range(num_vtm_blocks+1)
        ])
        # self.target_projection = nn.Sequential(
        #     nn.Linear(dims[-1], patch_dim//2),
        #     nn.GELU(),
        #     nn.Linear(patch_dim//2, patch_dim//2)
        # )
        self.main_tokens = []
        self.aux_tokens = []
        
    def compute_loss(self, main_tokens, aux_tokens):
        # loss = 0
        # B, N, D = self.source_main_tokens.shape
        # B, M, D = self.target_main_tokens.shape
        # # print(self.source_tokens.shape, self.target_tokens.shape)
        # C1 = torch.cdist(self.source_main_tokens, self.source_main_tokens).pow(2)
        # C2 = torch.cdist(self.target_main_tokens, self.target_main_tokens).pow(2)
        # M = torch.cdist(self.source_main_tokens, self.target_main_tokens).pow(2)
        # G_main = ot.solve_gromov_batch(C1, C2, M=M, alpha=0.5)

        # C1 = torch.cdist(self.source_aux_tokens, self.source_aux_tokens).pow(2)
        # C2 = torch.cdist(self.target_aux_tokens, self.target_aux_tokens).pow(2)
        # M = torch.cdist(self.source_aux_tokens, self.target_aux_tokens).pow(2)
        # G_aux = ot.solve_gromov_batch(C1, C2, M=M, alpha=0.5)
        # return G_main.value.mean() + G_aux.value.mean()
        loss = 0
        for i in range(len(main_tokens)-1):
            loss += self.compute_OT(main_tokens[0], main_tokens[i+1])
        for i in range(len(aux_tokens)-1):
            loss += self.compute_OT(aux_tokens[0], aux_tokens[i+1])
        return loss

    def compute_OT(self, src, tgt):
        C1 = torch.cdist(src, src).pow(2)
        C2 = torch.cdist(tgt, tgt).pow(2)
        # M = (C1.mean(dim=-1).unsqueeze(-1) - C2.mean(dim=-1).unsqueeze(-2)).pow(2)
        M = torch.cdist(src, tgt).pow(2)
        # print(C1.shape, C2.shape, M.shape)
        # exit()
        G = ot.solve_gromov_batch(C1, C2, M=M, alpha=0.5, grad = 'envelope')
        return G.value.mean()
    def forward(self, tokens: torch.Tensor):
        # tokens: (batch_size, num_tokens, patch_dim)
        # tokens = tokens + self.pos_embedding[:, :tokens.size(1)]
        main_tokens_list = []
        aux_tokens_list = []
        if self.training:
            main_tokens_list.append(self.projectors[0](tokens))
            aux_tokens_list.append(self.projectors[0](tokens))
            # main_tokens_list.append(tokens)
            # aux_tokens_list.append(tokens)
        aux_tokens = tokens.clone()
        for i, block in enumerate(self.vtm_blocks):
            tokens, aux_tokens = block(tokens, aux_tokens)
            if self.training:
                main_tokens_list.append(self.projectors[i+1](tokens))
                aux_tokens_list.append(self.projectors[i+1](aux_tokens))
                # main_tokens_list.append(tokens)
                # aux_tokens_list.append(aux_tokens)
        if self.training:
            final_representation1 = tokens.mean(dim=1)
            output = self.prediction_head1(final_representation1)
            final_representation2 = aux_tokens.mean(dim=1)
            aux_output = self.prediction_head2(final_representation2)
            return output, aux_output, self.compute_loss(main_tokens_list, aux_tokens_list)
        else:
            final_representation = tokens.mean(dim=1)
            output = self.prediction_head1(final_representation)
            return output