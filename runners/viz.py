import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your specific modules
from models.model_lvu import VideoTokenMergingTransformer
from datasets.breakfast_dataset import CustomDataset 

def generate_random_colors(N, seed=42):
    """Generate distinct random colors for contours."""
    np.random.seed(seed)
    colors = []
    for _ in range(N):
        # Generate bright, distinct colors
        colors.append((np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)))
    return colors

def features_to_rgb_pca(features_tensor, target_size=(224, 224)):
    """
    Convert (H, W, C) features to a viewable RGB image using PCA.
    Args:
        features_tensor: tensor of shape (H, W, C)
    """
    H, W, C = features_tensor.shape
    flat_feats = features_tensor.reshape(-1, C).cpu().numpy()
    
    # PCA to reduce to 3 channels (RGB)
    pca = PCA(n_components=3)
    rgb_flat = pca.fit_transform(flat_feats)
    
    # Normalize to 0-255 min-max
    rgb_flat = (rgb_flat - rgb_flat.min()) / (rgb_flat.max() - rgb_flat.min() + 1e-8)
    rgb_flat = (rgb_flat * 255).astype(np.uint8)
    
    # Reshape back to grid
    rgb_img = rgb_flat.reshape(H, W, 3)
    
    # Upscale to target size using Nearest Neighbor to keep "pixelated" feature look
    # or Linear if you want it smooth. The user asked for "blocky", so Nearest is often good,
    # but for the background, Linear looks nicer.
    rgb_resized = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_NEAREST)
    return rgb_resized

def visualize_merging(model, loader, device, save_dir='./vis_results_original', num_samples=5):
    model.eval()
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Params for reshaping (adjust based on your specific dataset config)
    T, H_grid, W_grid = 64, 7, 7  
    
    count = 0
    
    print("Starting visualization...")
    
    with torch.no_grad():
        for batch_idx, (vid_ids, inputs, labels, frame_range) in enumerate(loader):
            inputs = inputs.to(device) # (B, 3136, 1024)
            
            # Forward pass to get maps
            outputs, maps = model(inputs, return_maps=True)
            # print(maps)
            
            # Iterate through batch
            for b in range(inputs.shape[0]):
                if count >= num_samples: return
                
                vid_id = str(vid_ids[b].item()) if isinstance(vid_ids[b], torch.Tensor) else str(vid_ids[b])
                
                # 1. Prepare Background Image (Feature Visualization)
                # Reshape (N, C) -> (T, H, W, C)
                raw_feats = inputs[b].reshape(T, H_grid, W_grid, -1)
                # Average over time to get a spatial heatmap of the video
                spatial_feats = raw_feats.mean(dim=0) # (7, 7, 1024)
                
                # Convert to RGB via PCA
                background_img = features_to_rgb_pca(spatial_feats, target_size=(224, 224))
                
                # 2. Process Token Maps
                # Current assignments: (N,) -> 0..N-1
                current_assignments = torch.arange(inputs.shape[1], device=device)
                
                # We will store images for the plot
                plot_images = [background_img]
                titles = ["Input Features (PCA)"]
                
                for i, blk_map in enumerate(maps):
                    # Update assignments
                    # blk_map is (B, N_prev) -> values are indices in N_next
                    m = blk_map[b] 
                    current_assignments = m[current_assignments]
                    
                    # 3. Create the Spatial Mask
                    # Reshape assignments to (T, H, W)
                    assign_grid = current_assignments.reshape(T, H_grid, W_grid)
                    
                    # Collapse Time: Take the MODE (most frequent) cluster ID for each spatial pixel
                    # This tells us: "This pixel usually belongs to Cluster X"
                    spatial_mask, _ = torch.mode(assign_grid, dim=0) # (7, 7)
                    spatial_mask = spatial_mask.cpu().numpy()
                    
                    # Resize mask to image size (Must use NEAREST to keep integer IDs valid)
                    mask_large = cv2.resize(spatial_mask.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST).astype(int)
                    
                    # 4. Draw Contours and Overlays
                    vis_img = background_img.copy()
                    overlay = np.zeros_like(vis_img)
                    
                    unique_ids = np.unique(mask_large)
                    colors = generate_random_colors(int(mask_large.max()) + 1)
                    
                    for uid in unique_ids:
                        # Binary mask for this specific cluster
                        cluster_mask = (mask_large == uid).astype(np.uint8)
                        
                        color = colors[uid]
                        
                        # A. Draw Contours (Thick lines)
                        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, color, thickness=2)
                        
                        # B. Semi-transparent fill
                        # We apply color to the overlay layer
                        overlay[cluster_mask == 1] = color

                    # Blend fill
                    alpha = 0.3
                    # Only blend where we have mask
                    mask_bool = np.any(overlay > 0, axis=-1)
                    vis_img[mask_bool] = cv2.addWeighted(vis_img[mask_bool], 1-alpha, overlay[mask_bool], alpha, 0).reshape(-1, 3)
                    
                    plot_images.append(vis_img)
                    titles.append(f"Block {i+1}\n({len(unique_ids)} clusters)")
                
                # 5. Save only final image
                print(frame_range)
                final_img = plot_images[-1]
                out_path = f'{save_dir}/vis_{count}_{vid_id}_{labels}.png'
                cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
                print(f"Saved vis_{count}_{vid_id}_{labels}.png")
                
                count += 1

if __name__ == "__main__":
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='/data/susimmuk/long-video/best_model_original_breakfast.pth')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 1. Load Data
    # Assuming you have your dataset args set up similarly to previous scripts
    # You might need to add arguments for root path etc.
    d_args = argparse.Namespace(
        dataset='Breakfast', 
        l_secs=64, 
        batch_size=1, 
        device=args.device
    )
    testset = CustomDataset(args=d_args, split='test')
    loader = DataLoader(testset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    model = VideoTokenMergingTransformer(num_classes=10, num_tokens=3136, patch_dim=1024).to(args.device)
    state = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(state, strict=False)
    
    # 3. Run
    visualize_merging(model, loader, args.device)