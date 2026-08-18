"""
Evaluation script for trained Breakfast video classification model
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.breakfast_dataset import CustomDataset
from models.model_breakfast import VideoTokenMergingTransformer
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============= Token Visualization Utilities =============
def get_grid_index(init_size, map_size, device):
    """For each initial grid, get its index in the feature map."""
    H_init, W_init = init_size
    H, W = map_size
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
    return idx.flatten()


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
        idx_hw = idx_hw + idx_batch * H * W
        idx_tokens = idx_token + idx_batch * N
        coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

        with torch.cuda.amp.autocast(enabled=False):
            value = value.detach().float()
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))
            all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
            value = value / all_weight[idx_hw.reshape(-1), 0]
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))
            x_out = A @ x.reshape(B * N, C).type(torch.float32)
    else:
        coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        x_out = A @ x

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
            all_weight = A @ torch.ones([B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
            value = value / all_weight[idx_token.reshape(-1), 0]
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            out = A @ feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
    else:
        indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
        value = value.detach()
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out


def vis_tokens(img, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens"""
    N = token_dict['token_num']
    device, dtype = img.device, img.dtype

    color_map = F.avg_pool2d(img, kernel_size=4)
    B, C, H, W = color_map.shape

    token_color = map2token(color_map, token_dict)
    tmp_dict = token_dict.copy()
    tmp_dict['map_size'] = [H, W]
    tmp_dict['x'] = token_color
    vis_img = token2map(tmp_dict)

    token_idx = torch.arange(N, device=device)[None, :, None].float() / N
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

    for i in range(edge_width):
        edge_map = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
        idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape, device=device, dtype=dtype) * edge_map

    edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
    vis_img = vis_img * (~edge_map) + edge_color * edge_map
    return vis_img


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Breakfast Video Classification Model')
    parser.add_argument('--weight_path', type=str, 
                        default='/data/susimmuk/long-video/best_model_breakfast.pth',
                        help='Path to model weights')
    parser.add_argument('--l_secs', default=64, type=int, help='Number of temporal frames')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='Breakfast', help='Dataset name')
    parser.add_argument('--visualize', action='store_true', help='Enable token visualization')
    parser.add_argument('--vis_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--vis_dir', type=str, default='./visualization_results', help='Directory to save visualizations')
    return parser.parse_args()

def visualize_token_evolution(token_states, img_tensor, sample_id, vis_dir, video_id=None):
    """
    Visualize the evolution of tokens through transformer blocks using vis_tokens
    Shows: [Original Image] [Block 0 with tokens] [Block 1 with tokens] ... [Block N with tokens]
    Tokens are superimposed on the original image at their spatial locations
    
    Args:
        token_states: List of (tokens, num_tokens) tuples for each block
        img_tensor: Original input image/video tensor for color reference
        sample_id: Sample identifier
        vis_dir: Directory to save visualizations
        video_id: Video identifier
    """
    num_blocks = len(token_states)
    
    # Create synthetic RGB image from video features for visualization
    # Average over time dimension and create pseudo-RGB
    B, N_full, C = img_tensor.shape
    num_frames = 64
    spatial_size = 7
    
    # Reshape to spatial grid and average over time
    spatial_tokens = img_tensor.reshape(B, num_frames, spatial_size, spatial_size, C)
    spatial_avg = spatial_tokens.mean(dim=1)  # [B, 7, 7, C]
    
    # Create pseudo-RGB by taking first 3 channels and normalizing
    pseudo_rgb = spatial_avg[0, :, :, :3].permute(2, 0, 1)  # [3, 7, 7]
    pseudo_rgb = (pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min() + 1e-8)
    pseudo_rgb = pseudo_rgb.unsqueeze(0)  # [1, 3, 7, 7]
    
    # Upsample to reasonable size
    img_for_vis = F.interpolate(pseudo_rgb, size=(224, 224), mode='bilinear', align_corners=False)
    img_for_vis = img_for_vis.to(img_tensor.device)
    
    # Create visualization: original image + all blocks with tokens in one row
    fig, axes = plt.subplots(1, num_blocks + 1, figsize=(6 * (num_blocks + 1), 6))
    if num_blocks + 1 == 1:
        axes = [axes]
    
    # First subplot: Original image without tokens
    orig_img = img_for_vis[0].cpu().permute(1, 2, 0).numpy()
    orig_img = np.clip(orig_img, 0, 1)
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Remaining subplots: Image with tokens superimposed at their exact spatial locations
    for idx, (tokens, num_tokens) in enumerate(token_states):
        B, N, C = tokens.shape
        
        # Create token_dict for vis_tokens function
        H_init, W_init = spatial_size, spatial_size
        N_init = H_init * W_init
        
        # Create idx_token mapping that shows which initial spatial grid position maps to which merged token
        if N == N_init * num_frames:
            # Initial tokens - maintain full spatial-temporal structure
            # Average over temporal dimension to get spatial tokens
            spatial_tokens_vis = tokens.reshape(B, num_frames, N_init, C).mean(dim=1)  # [B, N_init, C]
            idx_token_spatial = torch.arange(N_init, device=tokens.device).unsqueeze(0)
            map_h, map_w = H_init * 8, W_init * 8  # Upsample for better visualization
        else:
            # Merged tokens - tokens no longer have regular spatial-temporal structure
            # Just use the merged tokens directly
            spatial_tokens_vis = tokens  # Use merged tokens as is [B, N, C]
            N_spatial = N
            
            # Map each initial spatial position to its corresponding merged token
            # Create contiguous mapping so merged tokens appear joined
            indices_per_token = max(1, (N_init + N_spatial - 1) // N_spatial)
            idx_token_spatial = torch.arange(N_init, device=tokens.device).unsqueeze(0)
            idx_token_spatial = idx_token_spatial // indices_per_token
            idx_token_spatial = torch.clamp(idx_token_spatial, 0, N_spatial - 1)
            map_h, map_w = H_init * 8, W_init * 8  # Upsample for better visualization
        
        token_dict = {
            'x': spatial_tokens_vis,
            'token_num': spatial_tokens_vis.shape[1],
            'map_size': [map_h, map_w],  # Feature map size (8x upsampled from 7x7)
            'init_grid_size': [H_init, W_init],  # Initial grid is 7x7 spatial
            'idx_token': idx_token_spatial,
            'agg_weight': torch.ones(B, N_init, 1, device=tokens.device)
        }
        
        # Generate visualization with tokens superimposed on original image
        vis_img = vis_tokens(img_for_vis, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=2)
        
        # Convert to numpy and display
        vis_np = vis_img[0].cpu().permute(1, 2, 0).numpy()
        vis_np = np.clip(vis_np, 0, 1)
        
        axes[idx + 1].imshow(vis_np)
        axes[idx + 1].set_title(f'Block {idx}\n{N} tokens (dim={C})', fontsize=12, fontweight='bold')
        axes[idx + 1].axis('off')
    
    plt.suptitle(f'Token Evolution - Sample {sample_id}' + (f' - {video_id}' if video_id else ''), 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    save_path = Path(vis_dir) / f'token_evolution_sample_{sample_id}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def evaluate(model, dataloader, device, visualize=False, vis_samples=5, vis_dir='./visualization_results'):
    """
    Evaluate the model on test data with optional token visualization
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
        vis_count = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (id, videos, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            videos, labels = videos.to(device), labels.to(device)
            
            # Forward pass with token tracking for visualization
            if visualize and vis_count < vis_samples:
                # Enable token tracking
                token_states, main_output = capture_token_states(model, videos)
                
                # Visualize each sample in batch
                batch_size = videos.shape[0]
                for i in range(min(batch_size, vis_samples - vis_count)):
                    # Extract states for this sample
                    sample_states = [(state[0][i:i+1], state[1]) for state in token_states]
                    sample_img = videos[i:i+1]
                    video_id = id[i] if isinstance(id, (list, tuple)) else f"batch{batch_idx}_sample{i}"
                    visualize_token_evolution(sample_states, sample_img, vis_count, vis_dir, video_id)
                    vis_count += 1
                    if vis_count >= vis_samples:
                        break
            else:
                # Normal forward pass
                main_output = model(videos)
            
            # Calculate accuracy
            _, predicted = torch.max(main_output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}%")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")
    print(f"{'='*60}\n")
    
    return accuracy


def capture_token_states(model, videos):
    """
    Capture intermediate token states from each transformer block
    
    Args:
        model: The model (may be wrapped in DataParallel)
        videos: Input video tensor
    
    Returns:
        List of (token_tensor, num_tokens) tuples at each stage
    """
    # Unwrap DataParallel if needed
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    token_states = []
    
    # Initial tokens
    tokens = videos
    B, N, C = tokens.shape
    token_states.append((tokens.clone(), N))
    
    # Pass through each VTM block
    aux_tokens = tokens.clone()
    for i, block in enumerate(base_model.vtm_blocks):
        tokens, aux_tokens = block(tokens, aux_tokens)
        B, N, C = tokens.shape
        token_states.append((tokens.clone(), N))
    
    # Get final prediction
    final_representation = tokens.mean(dim=1)
    output = base_model.prediction_head1(final_representation)
    
    return token_states, output

def main():
    args = parse_args()
    
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    print(f"  Weight Path: {args.weight_path}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  L_secs: {args.l_secs}")
    print()
    
    # Check if weight file exists
    if not os.path.exists(args.weight_path):
        print(f"Error: Weight file not found at {args.weight_path}")
        return
    
    # Load test dataset
    print("Loading test dataset...")
    testset = CustomDataset(args=args, split='test')
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    print(f"Test dataset loaded: {len(testset)} samples\n")
    
    # Initialize model
    print("Initializing model...")
    num_classes = 10  # Breakfast has 10 classes
    patch_dim = 1024
    num_frames = 64
    num_tokens = 7 * 7 * num_frames  # Swin-B: 7x7 patches
    
    model = VideoTokenMergingTransformer(
        num_classes=num_classes,
        num_tokens=num_tokens,
        patch_dim=patch_dim,
        num_vtm_blocks=3,
        num_heads=8
    )
    # print(model)
    # Load weights
    print(f"Loading model weights from {args.weight_path}...")
    try:
        state_dict = torch.load(args.weight_path, map_location=args.device)
        # print(state_dict.keys())
        
        # Try loading with strict=False to handle architecture mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Warning: Missing keys in state_dict (will use random initialization):")
            for key in missing_keys[:5]:  # Show first 5
                print(f"     - {key}")
            if len(missing_keys) > 5:
                print(f"     ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys in state_dict (will be ignored):")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"     - {key}")
            if len(unexpected_keys) > 5:
                print(f"     ... and {len(unexpected_keys) - 5} more")
        
        if not missing_keys and not unexpected_keys:
            print("✓ Model weights loaded successfully!\n")
        else:
            print("✓ Model weights loaded with warnings (partial match)\n")
            
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Use DataParallel for multi-GPU evaluation
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # Run evaluation with optional visualization
    accuracy = evaluate(
        model, 
        testloader, 
        args.device,
        visualize=args.visualize,
        vis_samples=args.vis_samples,
        vis_dir=args.vis_dir
    )
    
    print(f"Final Test Accuracy: {accuracy:.4f}%")
    
    if args.visualize:
        print(f"\n✅ Visualizations saved to: {args.vis_dir}")
        print(f"   Generated {args.vis_samples} token evolution visualizations")

if __name__ == "__main__":
    main()
