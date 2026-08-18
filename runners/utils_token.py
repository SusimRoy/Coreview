import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
# from mmcv.utils import get_logger
# from mmcv.runner import _load_checkpoint, load_state_dict
import re

'''
Note:
    B: batch size
    N: token number
    C: channel number
    N_init: initial token number
    H_init: height of initial grid
    W_init: width of initial grid
    H: height of feature map
    W: width of feature map
'''

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI."""
    checkpoint = _load_checkpoint(filename, map_location, logger)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
        
    _ = load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='tcformer', log_file=log_file, log_level=log_level)
    return logger


def get_grid_index(init_size, map_size, device):
    """For each initial grid, get its index in the feature map."""
    H_init, W_init = init_size
    H, W = map_size
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
    return idx.flatten()


def index_points(points, idx):
    """Sample features following the index."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def token2map(token_dict):
    """Transform vision tokens to feature map."""
    x = token_dict['x']
    H, W = token_dict['map_size']
    H_init, W_init = token_dict['init_grid_size']
    idx_token = token_dict['idx_token']
    B, N, C = x.shape
    N_init = H_init * W_init
    device = x.device

    if N_init == N and N == H * W:
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    idx_hw = get_grid_index([H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    value = x.new_ones(B * N_init)

    if N_init < N * H * W:
        # Sparse matrix multiplication
        idx_hw = idx_hw + idx_batch * H * W
        idx_tokens = idx_token + idx_batch * N
        coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

        with torch.cuda.amp.autocast(enabled=False):
            value = value.detach().float()
            # Updated to use sparse_coo_tensor
            A = torch.sparse_coo_tensor(coor, value, (B * H * W, B * N))

            all_weight = torch.sparse.mm(A, x.new_ones(B * N, 1).float()) + 1e-6
            value = value / all_weight[idx_hw.reshape(-1), 0]

            A = torch.sparse_coo_tensor(coor, value, (B * H * W, B * N))
            x_out = torch.sparse.mm(A, x.reshape(B * N, C).float())

    else:
        # Dense matrix multiplication
        coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)
        # Updated to use sparse_coo_tensor
        A = torch.sparse_coo_tensor(coor, value, (B, H * W, N)).to_dense()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        x_out = torch.bmm(A, x)

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return x_out


def map2token(feature_map, token_dict):
    """Transform feature map to vision tokens."""
    idx_token = token_dict['idx_token']
    N = token_dict['token_num']
    H_init, W_init = token_dict['init_grid_size']
    N_init = H_init * W_init

    agg_weight = None 

    B, C, H, W = feature_map.shape
    device = feature_map.device

    if N_init == N and N == H * W:
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    idx_hw = get_grid_index([H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    
    if agg_weight is None:
        value = feature_map.new_ones(B * N_init)
    else:
        value = agg_weight.reshape(B * N_init).type(feature_map.dtype)

    if N_init < N * H * W:
        idx_token = idx_token + idx_batch * N
        idx_hw = idx_hw + idx_batch * H * W
        indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

        with torch.cuda.amp.autocast(enabled=False):
            value = value.detach().float()
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            
            all_weight = torch.sparse.mm(A, torch.ones([B * H * W, 1], device=device, dtype=torch.float32)) + 1e-6
            value = value / all_weight[idx_token.reshape(-1), 0]

            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            
            flat_map = feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
            out = torch.sparse.mm(A, flat_map)
    else:
        indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
        value = value.detach()
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.bmm(A, feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous())

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out


def token_downup(target_dict, source_dict):
    """Transform token features between different distributions."""
    x_s = source_dict['x']
    idx_token_s = source_dict['idx_token']
    idx_token_t = target_dict['idx_token']
    T = target_dict['token_num']
    B, S, C = x_s.shape
    N_init = idx_token_s.shape[1]

    weight = target_dict.get('agg_weight', None)
    if weight is None:
        weight = x_s.new_ones(B, N_init, 1)
    weight = weight.reshape(-1)

    if N_init < T * S:
        idx_token_t = idx_token_t + torch.arange(B, device=x_s.device)[:, None] * T
        idx_token_s = idx_token_s + torch.arange(B, device=x_s.device)[:, None] * S
        coor = torch.stack([idx_token_t, idx_token_s], dim=0).reshape(2, B * N_init)

        with torch.cuda.amp.autocast(enabled=False):
            weight = weight.float().detach()
            A = torch.sparse_coo_tensor(coor, weight, (B * T, B * S))
            
            all_weight = torch.sparse.mm(A, x_s.new_ones(B * S, 1).float()) + 1e-6
            weight = weight / all_weight[(idx_token_t).reshape(-1), 0]
            
            A = torch.sparse_coo_tensor(coor, weight, (B * T, B * S))
            x_out = torch.sparse.mm(A, x_s.reshape(B * S, C).float())
    else:
        idx_batch = torch.arange(B, device=x_s.device)[:, None].expand(B, N_init)
        coor = torch.stack([idx_batch, idx_token_t, idx_token_s], dim=0).reshape(3, B * N_init)
        weight = weight.detach()
        A = torch.sparse_coo_tensor(coor, weight, (B, T, S)).to_dense()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        x_out = torch.bmm(A, x_s)

    x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out


def get_token_density_map(token_dict):
    """
    Get token density map. 
    Includes fixes for division by zero and deprecated sparse syntax.
    """
    N = token_dict['token_num']
    idx_token = token_dict['idx_token']
    B, N_init = idx_token.shape
    device = idx_token.device
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)

    coor = torch.stack([idx_batch, idx_token], dim=0).reshape(2, B * N_init)
    tmp = torch.ones(B * N_init, device=device)
    
    # Use sparse_coo_tensor and ensure density calculation is robust
    counts = torch.sparse_coo_tensor(coor, tmp, (B, N)).to_dense()
    
    # FIX: Add epsilon to avoid division by zero for empty tokens
    token_density = 1.0 / (counts + 1e-6)
    
    tmp_dict = token_dict.copy()
    tmp_dict['x'] = token_density[..., None]
    density_map = token2map(tmp_dict)
    return density_map

# =================================================================
# FLOPs Counters and Clustering Utils
# =================================================================

def map2token_flops(N_init, C):
    return N_init * (2 + 1 + 1 + C)

def token2map_flops(N_init, C):
    return N_init * (2 + 1 + 1 + C)

def downup_flops(N_init, C):
    return N_init * (2 + 1 + 1 + C)

def cluster_and_merge_flops(num_tokens, dim, k):
    flops = 0
    flops += num_tokens * num_tokens * dim  # distance matrix
    flops += num_tokens * k                 # local density
    flops += num_tokens * num_tokens        # distance indicator
    flops += num_tokens * dim               # token merge
    return flops

def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim

def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    with torch.no_grad():
        x = token_dict['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

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

        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num

def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    
    if agg_weight is not None:
        agg_weight_new = agg_weight * weight_t
        agg_weight_new = agg_weight_new / (agg_weight_new.max(dim=1, keepdim=True)[0] + 1e-8)
    else:
        agg_weight_new = None

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['map_size'] = token_dict['map_size']
    out_dict['init_grid_size'] = token_dict['init_grid_size']
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    return out_dict

def vis_tokens(img, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens."""
    N = token_dict['token_num']
    device, dtype = img.device, img.dtype

    # Check for color map shape or calculate average
    if img.shape[1] == 3:
         color_map = F.avg_pool2d(img, kernel_size=4)
    else:
         # Fallback if input is not standard RGB image
         color_map = F.avg_pool2d(img.mean(dim=1, keepdim=True).repeat(1,3,1,1), kernel_size=4)
         
    B, C, H, W = color_map.shape

    token_color = map2token(color_map, token_dict)
    tmp_dict = token_dict.copy()
    tmp_dict['map_size'] = [H, W]
    tmp_dict['x'] = token_color
    vis_img = token2map(tmp_dict)

    token_idx = torch.arange(N, device=device)[None, :, None].float() / (N + 1e-6)
    tmp_dict['x'] = token_idx
    idx_map = token2map(tmp_dict) 

    vis_img = F.interpolate(vis_img, [H * 8, W * 8], mode='nearest')
    idx_map = F.interpolate(idx_map, [H * 8, W * 8], mode='nearest')

    kernel = idx_map.new_zeros([4, 1, 3, 3])
    kernel[:, :, 1, 1] = 1
    kernel[0, :, 0, 1] = -1
    kernel[1, :, 2, 1] = -1
    kernel[2, :, 1, 0] = -1
    kernel[3, :, 1, 2] = -1

    edge_map = torch.zeros_like(idx_map[:,0:1,:,:])
    for i in range(edge_width):
        curr_edge = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        curr_edge = (curr_edge != 0).max(dim=1, keepdim=True)[0]
        edge_map = torch.max(edge_map, curr_edge)

    edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
    vis_img = vis_img * (~edge_map.bool()) + edge_color * edge_map.bool()
    return vis_img