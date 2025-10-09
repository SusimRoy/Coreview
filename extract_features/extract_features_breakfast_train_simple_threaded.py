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
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# Suppress checkpoint loading messages
logging.getLogger('mmcv').setLevel(logging.ERROR)
logging.getLogger('mmaction').setLevel(logging.ERROR)

# Add the Video-Swin-Transformer path to sys.path
sys.path.append('/home/csgrad/susimmuk/long-video/Video-Swin-Transformer')

from mmaction.apis import init_recognizer

# Check available GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Available GPUs: {num_gpus}")

# Use the proper Video-Swin-Transformer model
config_file = '/home/csgrad/susimmuk/long-video/extract_features/mmaction2/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
checkpoint_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k_fixed.pth'

DATA_ROOT = '/data_local3/susimmuk'

def process_video(args):
    video_id, gpu_id = args
    # Use specific GPU for this thread
    device = f'cuda:{gpu_id}' if num_gpus > 0 else 'cpu'
    # Load model for each thread on specific GPU
    model = init_recognizer(config_file, checkpoint_file, device=device)
    dest_mean = f'{DATA_ROOT}/Breakfast2/{video_id}.npy'

    if not os.path.exists(dest_mean):
        if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
            return
        file = video_id.split('.')[0].replace('-', '/')
        file = f'{DATA_ROOT}/BreakfastII_15fps_qvga_sync/{file}.avi'
        clip = VideoFileClip(file)
        n_frames = int(clip.duration * clip.fps)
        n_segments = 512
        segment_length = 32

        if n_frames < (n_segments + segment_length):
            starts = [i for i in range(n_frames - segment_length)]
        else:
            step = (n_frames - segment_length) / float(n_segments)
            starts = np.arange(0, n_frames - segment_length, step=step)

        mean_features = []
        for start in starts:
            start = int(start)
            frames = []
            for i in range(start, start + segment_length):
                image = cv2.resize(clip.get_frame(i / clip.fps), (224, 224), interpolation=cv2.INTER_AREA)
                frames.append(image)
            frames = np.asarray(frames) / 255.0
            frames = torch.from_numpy(frames.transpose([3, 0, 1, 2])).float()  # [C, T, H, W]
            frames = torch.unsqueeze(frames, 0).to(device)  # [1, C, T, H, W]

            with torch.no_grad():
                # Use the extract_feat method from MMAction2
                features = model.extract_feat(frames)[0]  # This should return the backbone features
                features = torch.squeeze(features, 0).detach().cpu().numpy()  # Remove batch dimension

            mean = reduce(features, 'c t h w -> c h w', 'mean')
            mean = rearrange(mean, 'c h w-> (h w) c')
            mean_features.append(mean)

        mean_features = np.asarray(mean_features)
        np.save(dest_mean, mean_features)

all_ids = []
csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/train.csv'
with open(csv_file, 'r') as f:
    f.readline()
    for line in f:
        video_id = line.split(',')[0]
        all_ids.append(video_id)

print('total files', len(set(all_ids)))
random.shuffle(all_ids)

# Assign GPUs to videos in round-robin fashion
if num_gpus > 0:
    video_gpu_pairs = [(video_id, i % num_gpus) for i, video_id in enumerate(all_ids)]
    max_workers = min(32, num_gpus)  # 4 threads per GPU
else:
    video_gpu_pairs = [(video_id, 0) for video_id in all_ids]
    max_workers = 4

print(f"Using {max_workers} workers across {num_gpus if num_gpus > 0 else 1} device(s)")

# Process videos in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_video, video_gpu_pairs), total=len(video_gpu_pairs)))