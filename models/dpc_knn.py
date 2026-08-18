import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class DPCKNNTokenSelector(nn.Module):
    def __init__(self, k_neighbors: int = 7, distance_threshold_percentile: float = 2.0, 
                 kde_bandwidth: float = 0.5):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.distance_threshold_percentile = distance_threshold_percentile
        self.kde_bandwidth = kde_bandwidth
        self.min_cluster_num = 64
        
    def compute_distances(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized_tokens = F.normalize(tokens, p=2, dim=-1)
        similarity = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))
        distances = 1 - similarity 
        return distances
    
    def compute_local_density(self, distances: torch.Tensor) -> torch.Tensor:
        B, N, _ = distances.shape
        knn_distances, knn_indices = torch.topk(distances, k=self.k_neighbors + 1, largest=False, dim=-1)
        knn_distances = knn_distances[:, :, 1:]  # (B, N, k)
        density = (-(knn_distances ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
        
        return density
    
    def compute_kde_density(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute token density using Kernel Density Estimation (KDE).
        
        Args:
            tokens: Input tokens of shape (B, N, C)
            
        Returns:
            Density estimates for each token of shape (B, N)
        """
        B, N, C = tokens.shape
        densities = []
        
        # Process each batch separately
        for b in range(B):
            # Move to CPU and convert to numpy for sklearn
            batch_tokens = tokens[b].detach().cpu().numpy()
            
            # Fit KDE
            kde = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
            kde.fit(batch_tokens)
            
            # Get log density scores
            log_density = kde.score_samples(batch_tokens)
            
            # Convert to actual density and create tensor
            density = torch.from_numpy(np.exp(log_density)).to(tokens.device)
            densities.append(density)
            
        # Stack all batches
        return torch.stack(densities)
    
    def compute_distance_to_higher_density(self, distances: torch.Tensor, densities: torch.Tensor) -> torch.Tensor:

        B, N = densities.shape
        higher_density_mask = densities[:, None, :] > densities[:, :, None]
        large_value = distances.max() + 1.0
        masked_distances = torch.where(higher_density_mask, distances, large_value)
        min_distances, _ = masked_distances.min(dim=2)
        no_higher_density = ~higher_density_mask.any(dim=2)  # (B, N)
        max_distances = distances.max(dim=2)[0]  # (B, N)
        min_distances = torch.where(no_higher_density, max_distances, min_distances)
        
        return min_distances    
    
    def select_cluster_centers(self, densities: torch.Tensor, min_distances: torch.Tensor, 
                            target_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = densities.shape
        target_count = max(1, int(N * target_ratio))   
        decision_values = densities * min_distances
        _, top_indices = torch.topk(decision_values, k=target_count, dim=-1)
        
        selected_mask = torch.zeros_like(decision_values, dtype=torch.bool)
        selected_mask.scatter_(1, top_indices, True)
        
        return selected_mask, top_indices
    
    def compute_token_scores(self, tokens: torch.Tensor, distances: torch.Tensor, 
                           cluster_center_indices: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        num_centers = cluster_center_indices.shape[1]
        
        # Extract distances from all tokens to cluster centers
        # distances[b, i, j] gives distance from token i to token j in batch b
        batch_indices = torch.arange(B, device=tokens.device).unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
        token_indices = torch.arange(N, device=tokens.device).unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        center_indices = cluster_center_indices.unsqueeze(1)  # (B, 1, num_centers)
        token_to_center_distances = distances[batch_indices.expand(B, N, num_centers), token_indices.expand(B, N, num_centers), center_indices.expand(B, N, num_centers)]
        min_distances_to_centers, closest_center_idx = token_to_center_distances.min(dim=2)  # (B, N)

        scores = 1/(min_distances_to_centers + 1e-6)

        # Normalize scores to [0, 1] range for interpretability
        score_min = scores.min(dim=1, keepdim=True)[0]  # (B, 1)
        score_max = scores.max(dim=1, keepdim=True)[0]  # (B, 1)
        score_range = score_max - score_min
        normalized_scores = torch.where(score_range > 0,(scores - score_min) / score_range, torch.zeros_like(scores))
        
        return normalized_scores, closest_center_idx
    
    def compute_density_aware_scores(self, tokens: torch.Tensor, distances: torch.Tensor,
                                   densities: torch.Tensor, cluster_center_indices: torch.Tensor) -> torch.Tensor:
        distance_scores, closest_center_idx = self.compute_token_scores(tokens, distances, cluster_center_indices)
        
        density_min = densities.min(dim=1, keepdim=True)[0]  # (B, 1)
        density_max = densities.max(dim=1, keepdim=True)[0]  # (B, 1)
        density_range = density_max - density_min
        normalized_densities = torch.where(density_range > 0, (densities - density_min) / density_range, torch.zeros_like(densities))

        combined_scores = 0.6 * distance_scores + 0.4 * normalized_densities
        
        return combined_scores, closest_center_idx
    
    # def cluster_dpc_knn(self, x, k, token_mask=None, threshold=0.53):
    #         with torch.no_grad():
    #             B, N, C = x.shape

    #             dist_matrix = torch.cdist(x, x) / (C ** 0.5)  # C * C

    #             if token_mask is not None:
    #                 token_mask = token_mask > 0
    #                 dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

    #             dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)  # C * k

    #             density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # C
    #             density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6  # C

    #             if token_mask is not None:
    #                 density = density * token_mask

    #             mask = density[:,None, :] > density[:, :, None]  # C * C
    #             mask = mask.type(x.dtype)
    #             dist_max = dist_matrix.flatten(1).max(dim=-1)[0][None, None]  # C * C
    #             print(mask.shape, dist_matrix.shape, dist_max.shape)
    #             dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)  # 1 * C, 1 * C

    #             score = dist * density
    #             batch_indices, col_indices = torch.where(score > threshold)
                
    #             if col_indices.numel() == 0:
    #                 _, indices = torch.topk(score, k=self.min_cluster_num, dim=1)  # (B, min_cluster_num)
    #                 batch_indices = torch.arange(B, device=x.device).repeat_interleave(self.min_cluster_num)
    #                 col_indices = indices.reshape(-1)

    #             # obtain the index of the cluster that each token belongs to
    #             batch_indices, col_indices = torch.where(score > threshold)
    #             dist_matrix_selected = dist_matrix[batch_indices, col_indices, :]  # Get selected points for each batch
                
    #             # Get distances to all cluster centers
    #             centers_dist = dist_matrix[:, :, col_indices]  # Shape: [B, N, num_centers]
                
    #             # Assign each point to nearest cluster center
    #             idx_cluster = centers_dist.argmin(dim=2)  # Shape: [B, N]
                
    #             # Create a mask for cluster centers
    #             cluster_mask = torch.zeros((B, N), dtype=torch.bool, device=x.device)
    #             cluster_mask[batch_indices, col_indices] = True
                
    #             # For cluster centers, assign their own index
    #             center_indices = torch.arange(len(col_indices), device=x.device)
    #             idx_cluster = torch.where(cluster_mask, center_indices[idx_cluster], idx_cluster)
                
    #             # Get original indices for centers
    #             index_down = col_indices
                
    #         return index_down, idx_cluster, score

    def index_points(self, points, idx):
        """Sample features following the index.
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def cluster_dpc_knn(self, x, k, token_mask=None, threshold=0.53):
        """Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
        """
        with torch.no_grad():
            B, N, C = x.shape

            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=self.min_cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)

            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, self.min_cluster_num)
            idx_tmp = torch.arange(self.min_cluster_num, device=x.device)[None, :].expand(B, self.min_cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, self.min_cluster_num
    
    def forward(self, tokens: torch.Tensor, target_ratio: float = 0.3, 
                return_scores: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = tokens.shape
        distances = self.compute_distances(tokens)
        densities = self.compute_local_density(distances)
        min_distances = self.compute_distance_to_higher_density(distances, densities)
        selected_mask, cluster_center_indices = self.select_cluster_centers(densities, min_distances, target_ratio)
        num_selected = selected_mask.sum(dim=1)[0].item()  # Assuming same count per batch
        selected_tokens = tokens[selected_mask].reshape(B, num_selected, C)   
        if return_scores:
            token_scores, closest_center_idx = self.compute_density_aware_scores(tokens, distances, densities, cluster_center_indices)
            return selected_tokens, selected_mask, token_scores, closest_center_idx
        else:
            return selected_tokens, selected_mask, None
    
    def get_token_scores(self, tokens: torch.Tensor, target_ratio: float = 0.3) -> torch.Tensor:
        _, _, scores, closest_center_idx = self.forward(tokens, target_ratio, return_scores=True)
        return scores, closest_center_idx
    
    def find_dense_region_tokens(self, tokens: torch.Tensor, density_threshold_percentile: float = 75.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find tokens that belong to dense regions using KDE.
        
        Args:
            tokens: Input tokens of shape (B, N, C)
            density_threshold_percentile: Percentile threshold for considering a region as dense
            
        Returns:
            dense_mask: Boolean mask indicating which tokens are in dense regions
            density_scores: Normalized density scores for all tokens
        """
        # Compute KDE-based density
        density_scores = self.compute_kde_density(tokens)
        
        # Normalize density scores to [0, 1]
        min_density = density_scores.min(dim=1, keepdim=True)[0]
        max_density = density_scores.max(dim=1, keepdim=True)[0]
        density_range = max_density - min_density
        normalized_density = (density_scores - min_density) / density_range
        
        # Calculate threshold for each batch
        threshold = torch.tensor([np.percentile(batch_density.cpu().numpy(), density_threshold_percentile) 
                                for batch_density in normalized_density]).to(tokens.device)
        
        # Create mask for tokens in dense regions
        dense_mask = normalized_density >= threshold.unsqueeze(1)
        
        return dense_mask, normalized_density
        
    def visualize_density_2d(self, tokens: torch.Tensor, batch_idx: int = 0, method: str = 'pca', 
                            density_threshold_percentile: float = 75.0, save_path: str = None):
        """
        Visualize token density distribution in 2D using PCA or t-SNE.
        
        Args:
            tokens: Input tokens of shape (B, N, C)
            batch_idx: Which batch to visualize
            method: Dimensionality reduction method ('pca' or 'tsne')
            density_threshold_percentile: Threshold for dense regions
            save_path: Path to save the plot (if None, display instead)
        """
        # Get tokens and compute density for specified batch
        batch_tokens = tokens[batch_idx].detach().cpu().numpy()
        dense_mask, density_scores = self.find_dense_region_tokens(tokens, density_threshold_percentile)
        batch_density = density_scores[batch_idx].cpu().numpy()
        batch_mask = dense_mask[batch_idx].cpu().numpy()
        
        # Reduce dimensionality to 2D
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30)
        
        tokens_2d = reducer.fit_transform(batch_tokens)
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot 1: All tokens colored by density
        plt.subplot(121)
        scatter = plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=batch_density, 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Density Score')
        plt.title(f'Token Density Distribution ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Plot 2: Dense vs Non-dense regions
        plt.subplot(122)
        plt.scatter(tokens_2d[~batch_mask, 0], tokens_2d[~batch_mask, 1], 
                   c='lightgray', alpha=0.5, label='Non-dense')
        plt.scatter(tokens_2d[batch_mask, 0], tokens_2d[batch_mask, 1], 
                   c='red', alpha=0.6, label='Dense')
        plt.title(f'Dense Regions (>{density_threshold_percentile}th percentile)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def visualize_density_distribution(self, tokens: torch.Tensor, batch_idx: int = 0, 
                                     density_threshold_percentile: float = 75.0, save_path: str = None):
        """
        Visualize the distribution of density scores as a histogram.
        
        Args:
            tokens: Input tokens of shape (B, N, C)
            batch_idx: Which batch to visualize
            density_threshold_percentile: Threshold for dense regions
            save_path: Path to save the plot (if None, display instead)
        """
        # Compute density scores
        _, density_scores = self.find_dense_region_tokens(tokens, density_threshold_percentile)
        batch_density = density_scores[batch_idx].cpu().numpy()
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(batch_density, bins=50, alpha=0.75, color='blue')
        
        # Add threshold line
        threshold = np.percentile(batch_density, density_threshold_percentile)
        plt.axvline(x=threshold, color='red', linestyle='--', 
                   label=f'{density_threshold_percentile}th percentile')
        
        plt.title('Distribution of Token Density Scores')
        plt.xlabel('Density Score')
        plt.ylabel('Count')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()