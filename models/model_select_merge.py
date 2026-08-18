import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
# from models.dpc_knn import DPCKNNTokenMerger
from models.dpc_knn import DPCKNNTokenSelector
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.visualization import visualize_token_selection, visualize_token_merging
from models.utils import trunc_normal_  
import matplotlib.pyplot as plt
import numpy as np
from models.visualization import visualize_token

## workaround 
torch.inverse(torch.ones((0, 0), device="cuda:0"))

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
        # self.token_selector = DPCKNNTokenSelector(k_neighbors=k_neighbors)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=out_dim)
        self.alpha = 0.5
        self.gamma = gamma
        self.k_neighbours = k_neighbors
        self.threshold = 0.53
        self.score = nn.Linear(dim, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.H = 7
        self.W = 7
        self.num_object_queries = 64  # Number of object prototypes
        self.norm = nn.LayerNorm(dim)
        # self.object_queries = nn.Parameter(torch.randn(1, self.num_object_queries, dim))
        self.conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
    
    def info_maximize(self, I, K, iters, p):
        X = torch.ones_like(I) / I.size(1)
        for _ in range(iters):
            # Use batch matrix-vector multiplication instead of torch.dot
            KX = torch.bmm(K, X.unsqueeze(-1)).squeeze(-1)
            X = torch.softmax(p*I - 2*p*self.alpha*KX, dim=-1)
        return X
    
    def index_points(self, points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def cluster_dpc_knn(self, x, cluster_num, k_neighbors, token_mask=None):
        with torch.no_grad():
            B, N, C = x.shape
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)
            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k_neighbors, dim=-1, largest=False)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                density = density * token_mask
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)
            dist_matrix = self.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num

    def merge_tokens(self, token_dict, idx_cluster, num_clusters, token_weight=None):
        x = token_dict['x']
        idx_token = token_dict['idx_token']
        agg_weight = token_dict['agg_weight']

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, 1)

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * num_clusters

        all_weight = token_weight.new_zeros(B * num_clusters, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                            source=token_weight.reshape(B * N, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # average token features
        x_merged = x.new_zeros(B * num_clusters, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                            source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, num_clusters, C)

        idx_token_new = self.index_points(idx_cluster[..., None], idx_token).squeeze(-1)
        weight_t = self.index_points(norm_weight, idx_token)
        agg_weight_new = agg_weight * weight_t
        agg_weight_new = agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        out_dict = {}
        out_dict['x'] = x_merged
        out_dict['token_num'] = num_clusters
        out_dict['idx_token'] = idx_token_new
        out_dict['agg_weight'] = agg_weight_new
        return out_dict
  
    def forward(self, token_dict: dict, epoch: int=None, id: int=None, block_idx: int=None):
        x = token_dict['x']
        B, N, C = token_dict['x'].shape
        score = self.score(x)
        token_weight = score.exp()
        token_dict['token_score'] = score
        cluster_num = max(math.ceil(N * self.gamma), 1)
        idx_cluster, cluster_num = self.cluster_dpc_knn(x, cluster_num, self.k_neighbours)
        out_dict = self.merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        x = out_dict['x']
        x_norm = self.norm1(x)
        token_dict['x'] = self.norm1(token_dict['x'])
        x_prime, _ = self.attn1(x_norm, token_dict, out_dict)
        x_res = x + self.drop_path(x_prime)
        # x_res_norm = self.norm2(x_res)
        out_dict['x'] = self.mlp1(x_res)
        out_dict['token_score'] = score
        return out_dict, score
        # x = token_dict['x']
        # x_norm = self.norm1(x)
        # x_prime, _ = self.attn1(x_norm, token_dict, out_dict)
        # x_res = x + self.drop_path(x_prime)
        # return out_dict, score


class HybridVideoTokenMergingTransformer(nn.Module):
    def __init__(self, num_classes: int, num_tokens: int, patch_dim: int = 1024, 
                 num_vtm_blocks: int = 3, num_heads: int = 8, k_neighbors: int = 7, 
                 target_ratio: float = 0.3):
        super().__init__()

        dims = [patch_dim // (2**i) for i in range(num_vtm_blocks + 1)]  # [1024, 512, 256, 128]
        
        # Progressive reduction ratios
        target_ratios = [target_ratio *(0.5**i) for i in range(num_vtm_blocks)]
        target_ratios = [max(0.1, ratio) for ratio in target_ratios]
        k_neighbors = [5,5,3]
        drop_path_rates = [0.1, 0.1, 0.1]
        gamma = [0.25,0.25,0.25]
        
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
        if num_classes > 0:
            self.prediction_head1 = nn.Sequential(
                nn.Linear(final_dim, num_classes)
            )
        else:
            self.prediction_head1 = nn.Sequential(
                nn.Linear(final_dim, 1)
            )
        self.apply(self._init_weights)
        self.projectors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dims[i], patch_dim//2),
                        nn.GELU(),
                        nn.Linear(patch_dim//2, patch_dim//2)
                    ) for i in range(num_vtm_blocks+1)
                ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_object_saliency(self, x, K=2):
        with torch.no_grad():
            A = torch.bmm(x, x.transpose(1, 2))            # [B, N, N]

            diag = A.sum(dim=-1)                           # [B, N]
            diag = torch.clamp(diag, min=1e-12)            # avoid divide-by-zero
            D = torch.diag_embed(diag)                     # [B, N, N]

            D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(diag))  # [B, N, N]
            L = D_inv_sqrt @ (D - A) @ D_inv_sqrt          # [B, N, N]
            eigenvals, eigenvecs = torch.linalg.eigh(L)    # eigenvecs: [B, N, N]
 
            y1 = (eigenvecs[:, :, 1:2] > 0).long()                 # [B, N, 1]
            S_t = x*y1                   
            return S_t, y1

    def forward(self, tokens: torch.Tensor, epoch: int=None, id: int=None):
            B, N, C = tokens.shape

            # Randomly sample 80% of the tokens to reduce sequence length and memory usage.
            # if self.training:
            #     num_sample_tokens = int(N * 0.3)
            # else:
            #     num_sample_tokens = int(N * 0.3)
            # indices = torch.randperm(N, device=tokens.device)[:num_sample_tokens]
            # indices = indices.sort()[0]  # Sorting is good practice for temporal consistency
            # tokens = tokens[:, indices, :]
            # N = num_sample_tokens
            # if id is not None and id == 0:
            #     visualize_token(tokens, frame_idx=0, channel_idx=32, batch_idx=0, save_path='./token_epoch_{}.png'.format(99),
            #                 title=f"Frame 0, Channel 0 (Epoch {epoch})")
            # tokens = tokens[:, 1:]
            # print(tokens.shape)
            device = tokens.device
            idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
            agg_weight = tokens.new_ones(B, N, 1)
            token_list = []
            token_score_list = [] 
            scores_list = []
            token_dict = {'x': tokens,
                        'token_num': N,
                        'idx_token': idx_token,
                        'agg_weight': agg_weight}
            token_list.append(self.projectors[0](tokens))
            for idx, block in enumerate(self.vtm_blocks):
                token_dict, spectral_score = block(token_dict, epoch, id, idx)
                if self.training:
                    proj_token = self.projectors[idx+1](token_dict['x'])
                    token_list.append(proj_token)
                    scores_list.append(F.softmax(spectral_score.squeeze(-1).float(), dim=1))
                if 'token_score' in token_dict:
                    token_score_list.append(token_dict['token_score'].detach())
            final_representation = token_dict['x'].mean(dim=1)
            output = self.prediction_head1(final_representation)
            if self.training:
                return output, token_list, token_score_list, scores_list
            return output, token_score_list