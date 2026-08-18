"""
Evaluation and Density Visualization for Breakfast VTM Model.
Uses 'Shadow Execution' to track token merging indices from the provided model structure.
"""
import os
import sys
import argparse
from utils_token import get_token_density_map
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- IMPORTS ---
try:
    from datasets.breakfast_dataset import CustomDataset
    from models.model_lvu import VideoTokenMergingTransformer
except ImportError:
    print("Error: Could not import dataset or model. Run this script from the project root.")
    sys.exit(1)

try:
    # This is the utility file you saved from the previous step
    from utils_token import get_token_density_map
except ImportError:
    print("Error: utils_token.py not found. Please save the utility code from the previous response.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Density Maps for VTM Model')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--vis_dir', type=str, default='./density_visualization', help='Output folder')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of videos to process')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset Arguments
    parser.add_argument('--dataset', type=str, default='Breakfast')
    parser.add_argument('--batch_size', type=int, default=1, help='Keep 1 for visualization')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--l_secs', default=64, type=int) 
    return parser.parse_args()


def save_heatmap(density_tensor, save_path, block_idx, token_count):
    """Saves a clean heatmap image without axes."""
    # density_tensor: [1, H, W] -> [H, W]
    if density_tensor.dim() == 4: density_tensor = density_tensor.squeeze()
    if density_tensor.dim() == 3: density_tensor = density_tensor.squeeze()
    
    map_np = density_tensor.cpu().numpy()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(map_np, cmap='inferno', interpolation='nearest')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def shadow_forward_pass(model, x):
    """
    Manually executes the logic of the VTM blocks to capture 'match_indices'.
    """
    model.eval()
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # --- HANDLING INPUT SHAPE ---
    # Case 1: Raw Video (B, C, T, H, W)
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
    # Case 2: Features (B, N, C) - ALREADY FLATTENED
    elif x.dim() == 3:
        pass # Already in correct shape
    
    current_tokens = x
    
    # Track the mapping from Initial Grid -> Current Token Index
    B, N_total, C = current_tokens.shape
    
    # Initial Map: [B, N_init] containing 0..N_init-1
    token_map = torch.arange(N_total, device=x.device).unsqueeze(0).expand(B, -1)
    
    visualization_data = []
    
    # Record Initial State
    visualization_data.append({
        'block_idx': -1,
        'tokens': current_tokens,
        'token_map': token_map.clone(), 
        'num_tokens': N_total
    })

    # Iterate Blocks
    for i, block in enumerate(base_model.vtm_blocks):
        # A. Main Path Normalization
        x_norm = block.norm1(current_tokens)
        
        # B. Attention and K matrix
        x_prime, K_matrix = block.attn1(x_norm)
        x_res = current_tokens + x_prime
        
        # C. Saliency and Sampling
        saliency_scores = torch.tanh(block.saliency_head(K_matrix)) 
        B, N, _ = current_tokens.shape
        num_targets = N // block.gamma
        
        sampling_probs = F.softmax(saliency_scores.squeeze(-1), dim=1)
        target_indices = torch.multinomial(sampling_probs, num_samples=num_targets, replacement=False)
        
        target_mask = torch.zeros_like(sampling_probs, dtype=torch.bool).scatter_(1, target_indices, True)
        source_mask = ~target_mask
        
        # D. Split Tokens
        source_tokens = x_res[source_mask].reshape(B, -1, C)
        target_tokens = x_res[target_mask].reshape(B, num_targets, C)
        source_keys = K_matrix[source_mask].reshape(B, -1, C)
        target_keys = K_matrix[target_mask].reshape(B, num_targets, C)
        
        # E. Matching Logic
        source_keys_norm = F.normalize(source_keys, p=2, dim=-1)
        target_keys_norm = F.normalize(target_keys, p=2, dim=-1)
        similarity = torch.bmm(source_keys_norm, target_keys_norm.transpose(1, 2))
        match_indices = similarity.argmax(dim=2)
        
        # F. Merging
        merged_tokens = target_tokens.clone()
        match_indices_expanded = match_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_tokens.scatter_add_(1, match_indices_expanded, source_tokens)
        
        counts = torch.ones_like(target_tokens[:, :, 0])
        counts.scatter_add_(1, match_indices, torch.ones_like(match_indices, dtype=torch.float))
        
        merged_main = merged_tokens / counts.unsqueeze(-1)
        merged = block.mlp1(merged_main)
        
        # UPDATE VISUALIZATION MAPPING
        old_to_new = torch.zeros((B, N), dtype=torch.long, device=x.device)
        new_target_ids = torch.arange(num_targets, device=x.device).unsqueeze(0).expand(B, -1)
        old_to_new.scatter_(1, target_indices, new_target_ids)
        old_to_new[source_mask] = match_indices.flatten() 
        token_map = torch.gather(old_to_new, 1, token_map)
        
        visualization_data.append({
            'block_idx': i,
            'tokens': merged,
            'token_map': token_map.clone(),
            'num_tokens': num_targets
        })
        
        current_tokens = merged
        
    return visualization_data


def run_evaluation(args):
    device = torch.device(args.device)
    save_dir = Path(args.vis_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.dataset} Dataset...")
    test_set = CustomDataset(args=args, split='test')
    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print("Loading Model...")
    model = VideoTokenMergingTransformer(
        num_classes=10, 
        num_tokens=7*7*64, 
        patch_dim=1024, 
        num_vtm_blocks=3,
        num_heads=8
    ).to(device)
    
    if os.path.exists(args.weight_path):
        state = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Weights loaded successfully.")
    else:
        print(f"Warning: Weights not found at {args.weight_path}")

    print(f"Generating density maps for {args.num_samples} samples...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            if batch_idx >= args.num_samples: break
            
            # Flexible unpacking for different dataset returns
            if len(batch_data) == 3:
                vid_id, videos, labels = batch_data
            else:
                videos, labels = batch_data
                vid_id = batch_idx

            videos = videos.to(device)
            
            # Setup Sample Directory
            if isinstance(vid_id, (list, tuple, torch.Tensor)): 
                if isinstance(vid_id, torch.Tensor) and vid_id.numel() > 1:
                    vid_id_str = str(vid_id[0].item())
                else:
                    vid_id_str = str(vid_id[0])
            else: 
                vid_id_str = str(vid_id)
            
            sample_dir = save_dir / f"sample_{vid_id_str}"
            sample_dir.mkdir(exist_ok=True)
            
            # --- FIX: Handling 3D Input (Features) ---
            if videos.dim() == 5:
                # Raw Video: Save Reference Frame
                mid_t = videos.shape[2] // 2
                ref_img = videos[0, :, mid_t, :, :].cpu().permute(1, 2, 0).numpy()
                ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
                plt.imsave(sample_dir / "input_frame.png", ref_img)
            else:
                # Features: Cannot save raw image. 
                # Create a "Feature Norm" heatmap to act as the reference.
                feature_norm = videos[0].norm(dim=1) # (N,)
                # Attempt to reshape to 7x7 spatial if possible
                N = feature_norm.shape[0]
                T = 64 # Assumed from args
                spatial_h, spatial_w = 7, 7
                if N == T * spatial_h * spatial_w:
                    feat_vis = feature_norm.reshape(T, spatial_h, spatial_w).mean(0).cpu().numpy()
                    plt.imsave(sample_dir / "input_feature_norm.png", feat_vis, cmap='viridis')

            # Run Shadow Forward Pass
            vis_states = shadow_forward_pass(model, videos)
            
            # Assuming 7x7 spatial structure for visualization
            spatial_h, spatial_w = 7, 7 
            
            for state in vis_states:
                idx = state['block_idx']
                token_map = state['token_map'] 
                num_tokens = state['num_tokens']
                
                token_dict = {
                    'x': torch.ones(1, num_tokens, 1, device=device), 
                    'token_num': num_tokens,
                    'idx_token': token_map, 
                    'map_size': [spatial_h*8, spatial_w*8], 
                    'init_grid_size': [spatial_h, spatial_w],
                    'agg_weight': None
                }
                
                try:
                    density_map = get_token_density_map(token_dict)
                    save_name = sample_dir / f"block_{idx+1}_tokens_{num_tokens}.png"
                    save_heatmap(density_map[0], save_name, idx, num_tokens)
                except Exception as e:
                    pass

    print(f"Done. Check {args.vis_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)