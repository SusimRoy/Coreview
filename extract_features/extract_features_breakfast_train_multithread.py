import torch
import os
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import *
import cv2
import numpy as np
from einops import rearrange, reduce
from tqdm import tqdm
import sys
import threading
from queue import Queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add the Video-Swin-Transformer path to sys.path
sys.path.append('/home/csgrad/susimmuk/long-video/Video-Swin-Transformer')

from mmaction.apis import init_recognizer

def load_model(config_file, checkpoint_file, device):
    """Load the model for feature extraction"""
    model = init_recognizer(config_file, checkpoint_file, device=device)
    return model

def extract_frames_from_video(video_path, starts, segment_length):
    """Extract frames from video for given start positions"""
    try:
        clip = VideoFileClip(video_path)
        all_frame_batches = []
        
        for start in starts:
            start = int(start)
            frames = []
            for i in range(start, start + segment_length):
                try:
                    image = cv2.resize(clip.get_frame(i / clip.fps), (224, 224), interpolation=cv2.INTER_AREA)
                    frames.append(image)
                except Exception as e:
                    # If we can't get a frame, duplicate the last one or use zeros
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            frames = np.asarray(frames) / 255.0
            frames = torch.from_numpy(frames.transpose([3, 0, 1, 2])).float()  # [C, T, H, W]
            all_frame_batches.append(frames)
        
        clip.close()
        return all_frame_batches
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def process_single_video(args):
    """Process a single video and extract features"""
    video_id, model, device, DATA_ROOT = args
    
    dest_mean = f'{DATA_ROOT}/Breakfast2/{video_id}.npy'
    
    # Skip if already processed
    if os.path.exists(dest_mean):
        return f"Skipped {video_id} (already exists)"
    
    # Skip problematic videos
    if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
        return f"Skipped {video_id} (problematic)"
    
    try:
        file = video_id.split('.')[0].replace('-', '/')
        file = f'{DATA_ROOT}/BreakfastII_15fps_qvga_sync/{file}.avi'
        
        # Check if video file exists
        if not os.path.exists(file):
            return f"Skipped {video_id} (file not found)"
        
        # Get video info
        clip = VideoFileClip(file)
        n_frames = int(clip.duration * clip.fps)
        n_segments = 512
        segment_length = 32
        clip.close()
        
        # Calculate start positions
        if n_frames < (n_segments + segment_length):
            starts = [i for i in range(max(1, n_frames - segment_length))]
        else:
            step = (n_frames - segment_length) / float(n_segments)
            starts = np.arange(0, n_frames - segment_length, step=step)
        
        # Extract frames
        frame_batches = extract_frames_from_video(file, starts, segment_length)
        if frame_batches is None:
            return f"Failed to extract frames from {video_id}"
        
        # Process frames through model
        mean_features = []
        for frames in frame_batches:
            frames = torch.unsqueeze(frames, 0).to(device)  # [1, C, T, H, W]
            
            with torch.no_grad():
                try:
                    # Use the extract_feat method from MMAction2
                    features = model.extract_feat(frames)[0]  # This should return the backbone features
                    features = torch.squeeze(features, 0).detach().cpu().numpy()  # Remove batch dimension
                    
                    mean = reduce(features, 'c t h w -> c h w', 'mean')
                    mean = rearrange(mean, 'c h w-> (h w) c')
                    mean_features.append(mean)
                except Exception as e:
                    print(f"Error processing frames for {video_id}: {e}")
                    continue
        
        if mean_features:
            mean_features = np.asarray(mean_features)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(dest_mean), exist_ok=True)
            np.save(dest_mean, mean_features)
            return f"Processed {video_id} successfully ({mean_features.shape})"
        else:
            return f"Failed to process {video_id} (no features extracted)"
            
    except Exception as e:
        return f"Error processing {video_id}: {e}"

def worker_process(video_ids, worker_id, config_file, checkpoint_file, DATA_ROOT, results_queue):
    """Worker process function"""
    try:
        # Each worker gets its own GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = f'cuda:{worker_id % torch.cuda.device_count()}'
        
        # Load model in worker process
        model = load_model(config_file, checkpoint_file, device)
        
        # Process videos
        for video_id in tqdm(video_ids, desc=f"Worker {worker_id}", position=worker_id):
            result = process_single_video((video_id, model, device, DATA_ROOT))
            results_queue.put(result)
            
    except Exception as e:
        results_queue.put(f"Worker {worker_id} error: {e}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_file = '/home/csgrad/susimmuk/long-video/extract_features/mmaction2/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
    checkpoint_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k_fixed.pth'
    DATA_ROOT = '/data_local3/susimmuk'
    
    # Load video IDs
    all_ids = []
    csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/train.csv'
    with open(csv_file, 'r') as f:
        f.readline()
        for line in f:
            video_id = line.split(',')[0]
            all_ids.append(video_id)
    
    print(f'Total files: {len(set(all_ids))}')
    all_ids = list(set(all_ids))  # Remove duplicates
    random.shuffle(all_ids)
    
    # Filter out already processed files
    unprocessed_ids = []
    for video_id in all_ids:
        dest_mean = f'{DATA_ROOT}/Breakfast2/{video_id}.npy'
        if not os.path.exists(dest_mean):
            unprocessed_ids.append(video_id)
    
    print(f'Unprocessed files: {len(unprocessed_ids)}')
    
    if not unprocessed_ids:
        print("All files already processed!")
        return
    
    # Determine number of workers
    if torch.cuda.is_available():
        num_workers = min(torch.cuda.device_count(), 4)  # Limit to avoid memory issues
    else:
        num_workers = min(mp.cpu_count() // 2, 4)  # Use half the CPU cores
    
    print(f'Using {num_workers} workers')
    
    # Split work among workers
    chunk_size = len(unprocessed_ids) // num_workers
    worker_video_lists = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        if i == num_workers - 1:  # Last worker gets remaining videos
            end_idx = len(unprocessed_ids)
        else:
            end_idx = (i + 1) * chunk_size
        worker_video_lists.append(unprocessed_ids[start_idx:end_idx])
    
    # Create results queue
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Start worker processes
    processes = []
    for worker_id in range(num_workers):
        if worker_video_lists[worker_id]:  # Only start if there are videos to process
            p = mp.Process(
                target=worker_process,
                args=(worker_video_lists[worker_id], worker_id, config_file, checkpoint_file, DATA_ROOT, results_queue)
            )
            p.start()
            processes.append(p)
    
    # Monitor progress
    total_videos = len(unprocessed_ids)
    processed_count = 0
    start_time = time.time()
    
    try:
        while processed_count < total_videos:
            try:
                result = results_queue.get(timeout=60)  # 60 second timeout
                processed_count += 1
                print(f"[{processed_count}/{total_videos}] {result}")
                
                # Print progress every 10 videos
                if processed_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_video = elapsed_time / processed_count
                    remaining_videos = total_videos - processed_count
                    estimated_remaining_time = avg_time_per_video * remaining_videos
                    print(f"Progress: {processed_count}/{total_videos}, "
                          f"Avg time per video: {avg_time_per_video:.2f}s, "
                          f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                    
            except Exception as e:
                print(f"Error getting result from queue: {e}")
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=10)  # 10 second timeout
        if p.is_alive():
            print(f"Terminating process {p.pid}")
            p.terminate()
            p.join()
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print(f"Average time per video: {total_time/processed_count:.2f} seconds")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()