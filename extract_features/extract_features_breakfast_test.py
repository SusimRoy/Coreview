# import torch
# import os
# import random
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from moviepy.editor import *
# import cv2
# import numpy as np
# from mmaction.apis import init_recognizer, inference_recognizer
# # from mmaction.models import build_model
# from einops import rearrange, reduce, repeat
# from tqdm import tqdm

# config_file = '/home/csgrad/susimmuk/long-video/extract_features/mmaction2/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py'
# checkpoint_file = 'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = init_recognizer(config_file, checkpoint_file, device=device)

# DATA_ROOT = '/data_local3/susimmuk'

# all_ids = []
# csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/test.csv'
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
#         file = video_id.split('.')[0].replace('-', '/')
#         file = f'{DATA_ROOT}/BreakfastII_15fps_qvga_sync/{file}.avi'
#         clip = VideoFileClip(file)
#         n_frames = int(clip.duration * clip.fps)
#         n_segments = 64
#         segment_length = 32
#         n_required = int(n_segments * segment_length)

#         if n_frames < n_required:
#             step = (n_frames - segment_length) / float(n_segments)
#             starts = np.arange(0, n_frames - segment_length, step=step)
#         else:
#             step = n_frames / float(n_segments)
#             starts = np.arange(0, n_frames, step=step)
#         # print(cnt, file, n_frames, len(starts))

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
from transformers import SwinConfig, SwinModel
import torch
from tqdm import tqdm
from mmaction.apis import init_recognizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_file = '/home/csgrad/susimmuk/long-video/extract_features/mmaction2/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
checkpoint_file = '/home/csgrad/susimmuk/long-video/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k_fixed.pth'
model = init_recognizer(config_file, checkpoint_file, device=device)

DATA_ROOT = '/data_local3/susimmuk'

all_ids = []
csv_file = f'/home/csgrad/susimmuk/long-video/data/Breakfast/test.csv'
with open(csv_file, 'r') as f:
    f.readline()
    for line in f:
        video_id = line.split(',')[0]
        all_ids.append(video_id)

print('total files', len(set(all_ids)))
random.shuffle(all_ids)

for cnt, video_id in tqdm(enumerate(tqdm(all_ids))):
    dest_mean = f'{DATA_ROOT}/Breakfast2/{video_id}.npy'
    if not os.path.exists(dest_mean):
        file = video_id.split('.')[0].replace('-', '/')
        file = f'{DATA_ROOT}/BreakfastII_15fps_qvga_sync/{file}.avi'
        clip = VideoFileClip(file)
        n_frames = int(clip.duration * clip.fps)
        n_segments = 64
        segment_length = 32
        n_required = int(n_segments * segment_length)

        if n_frames < n_required:
            step = (n_frames - segment_length) / float(n_segments)
            starts = np.arange(0, n_frames - segment_length, step=step)
        else:
            step = n_frames / float(n_segments)
            starts = np.arange(0, n_frames, step=step)

        if n_frames < (n_segments+segment_length):
            starts = [i for i in range(n_frames-segment_length)]
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
