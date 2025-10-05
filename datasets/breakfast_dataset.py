import torch
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import cv2
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(1112)

DATA_ROOT = '/data_local3/susimmuk'

class CustomDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.videos = []
        self.labels = []
        csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/{split}.csv'
        with open(csv_file, 'r') as f:
            f.readline()
            for line in f:
                video_id = line.split(',')[0]
                if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
                    continue
                label = int(line.split(',')[-1])
                self.videos.append(video_id)
                self.labels.append(label)
            print('Total videos in ', split, len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_features = np.load(f'{DATA_ROOT}/lvu/{self.videos[idx]}.npy')

        if video_features.shape[0] < self.args.l_secs:
            step = video_features.shape[0] / float(self.args.l_secs)
            indices = np.arange(0, video_features.shape[0], step, dtype=np.float32).astype(np.int32)
            video_features = video_features[indices]

        elif video_features.shape[0] > self.args.l_secs:
            if self.split == 'train':
                indices = random.sample(range(0, video_features.shape[0]), self.args.l_secs)
                indices.sort()
                video_features = video_features[indices]
            else:
                step = video_features.shape[0] / float(self.args.l_secs)
                indices = np.arange(0, video_features.shape[0], step, dtype=np.float32).astype(np.int32)
                video_features = video_features[indices]

        video_features = np.reshape(video_features,(video_features.shape[0]* video_features.shape[1], 1024))

        return self.videos[idx], video_features, self.labels[idx]
    
    # def __getitem__(self, idx):
    #     # Construct the video file path
    #     file = self.videos[idx].split('.')[0].replace('-', '/')
    #     video_path = f'{DATA_ROOT}/BreakfastII_15fps_qvga_sync/{file}.avi'

    #     # Load video frames using moviepy
    #     from moviepy.editor import VideoFileClip
    #     from PIL import Image
    #     clip = VideoFileClip(video_path)
    #     total_frames = int(clip.fps * clip.duration)
    #     n_frames = 64
    #     # Uniformly sample 64 frame indices
    #     if total_frames >= n_frames:
    #         indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    #     else:
    #         # If not enough frames, repeat last frame
    #         indices = np.concatenate([
    #             np.linspace(0, total_frames - 1, total_frames, dtype=int),
    #             np.full(n_frames - total_frames, total_frames - 1, dtype=int)
    #         ])
    #     frames = []
    #     for i in indices:
    #         frame = clip.get_frame(i / clip.fps)
    #         # Resize to 224x224
    #         img = Image.fromarray(frame)
    #         img = img.resize((224, 224), Image.BILINEAR)
    #         frames.append(np.array(img))
    #     clip.close()
    #     video_array = np.stack(frames)  # (64, 224, 224, 3)
    #     video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()  # (64, 3, 224, 224)

    #     return self.videos[idx], video_tensor, self.labels[idx]

