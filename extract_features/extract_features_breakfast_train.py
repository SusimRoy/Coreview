# import torch
# import os
# import random
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from moviepy.editor import *
# import cv2
# import numpy as np
# from tqdm import tqdm
# from mmaction.apis import init_recognizer, inference_recognizer
# # from mmaction.models import build_model
# from einops import rearrange, reduce, repeat

# config_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py'
# checkpoint_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics400_22k.pth'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = init_recognizer(config_file, checkpoint_file, device=device)

# DATA_ROOT = '/data_local3/susimmuk'

# all_ids = []
# csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/train.csv'
# with open(csv_file, 'r') as f:
#     f.readline()
#     for line in f:
#         video_id = line.split(',')[0]
#         all_ids.append(video_id)

# print('total files', len(set(all_ids)))
# random.shuffle(all_ids)

# for cnt, video_id in tqdm(enumerate(all_ids)):
#     dest_mean = f'{DATA_ROOT}/lvu/{video_id}.npy'

#     if not os.path.exists(dest_mean):
#         if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
#             continue
#         file = video_id.split('.')[0].replace('-', '/')
#         file = f'{DATA_ROOT}/BreakfastII_15fps_qvga_sync/{file}.avi'
#         clip = VideoFileClip(file)
#         n_frames = int(clip.duration * clip.fps)
#         n_segments = 512
#         segment_length = 32

#         if n_frames < (n_segments+segment_length):
#             starts = [i for i in range(n_frames-segment_length)]
#         else:
#             step = (n_frames - segment_length) / float(n_segments)
#             starts = np.arange(0, n_frames - segment_length, step=step)

#         mean_features = []
#         for start in starts:
#             start = int(start)
#             # print(cnt, file, start, '/', n_frames)
#             frames = []
#             for i in range(start, start + segment_length):
#                 image = cv2.resize(clip.get_frame(i / clip.fps), (224, 224), interpolation=cv2.INTER_AREA)
#                 frames.append(image)
#             frames = np.asarray(frames) / 255.0
#             frames = torch.from_numpy(frames.transpose([3, 0, 1, 2])).float()
#             frames = torch.unsqueeze(frames, 0)
#             features = torch.squeeze(model.extract_feat(frames.to(device))[0]).detach().cpu().numpy()

#             mean = reduce(features, 'c t h w -> c h w', 'mean')
#             mean = rearrange(mean, 'c h w-> (h w) c')
#             mean_features.append(mean)

#         mean_features = np.asarray(mean_features)

#         # print(cnt, file)
#         # print(mean_features.shape)

#         np.save(dest_mean, mean_features)


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

# Add the Video-Swin-Transformer path to sys.path
sys.path.append('/home/csgrad/susimmuk/long-video/Video-Swin-Transformer')

from mmaction.apis import init_recognizer

# def fix_checkpoint_keys(state_dict):
#     """Fix key mismatches between Video Swin and standard Swin models"""
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key
#         # Map backbone.norm to backbone.norm3 (Video Swin -> Standard Swin)
#         if key == 'backbone.norm.weight':
#             new_key = 'backbone.norm3.weight'
#         elif key == 'backbone.norm.bias':
#             new_key = 'backbone.norm3.bias'
#         # Remove cls_head keys as they're not needed for feature extraction
#         # elif key.startswith('cls_head'):
#         #     continue
        
#         new_state_dict[new_key] = value
#     return new_state_dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the proper Video-Swin-Transformer model
config_file = '/home/csgrad/susimmuk/long-video/extract_features/mmaction2/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
checkpoint_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k_fixed.pth'
# ckpt = torch.load(checkpoint_file, map_location=device)
# fixed_state_dict = fix_checkpoint_keys(ckpt["state_dict"])
# fixed_checkpoint_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k_fixed.pth'
# fixed_ckpt = {
#     'state_dict': fixed_state_dict,
#     'meta': ckpt.get('meta', {}),  # Keep original metadata if it exists
# }
# torch.save(fixed_ckpt, fixed_checkpoint_file)
# print(fixed_state_dict.keys())
# use this fixed_state_dict to load the model by saving as a torch.save

model = init_recognizer(config_file, checkpoint_file, device=device)
# model.load_state_dict(fixed_state_dict, strict=False)

DATA_ROOT = '/data_local3/susimmuk'

all_ids = []
csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/train.csv'
with open(csv_file, 'r') as f:
    f.readline()
    for line in f:
        video_id = line.split(',')[0]
        all_ids.append(video_id)

print('total files', len(set(all_ids)))
random.shuffle(all_ids)

for cnt, video_id in enumerate(tqdm(all_ids)):
    dest_mean = f'{DATA_ROOT}/Breakfast2/{video_id}.npy'

    if not os.path.exists(dest_mean):
        if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
            continue
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
            # print(cnt, file, start, '/', n_frames)
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
        # print(cnt, file)
        # print(mean_features.shape)

        np.save(dest_mean, mean_features)
