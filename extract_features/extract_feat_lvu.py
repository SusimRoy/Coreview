import timm
import numpy as np
import pandas as pd
import skvideo.io
import cv2
import torch
import os
import glob
import torch.nn as nn
import random
import pickle
from tqdm import tqdm 

DATA_ROOT = '/data_local3/susimmuk/'

duration_data = pd.read_csv('/home/csgrad/susimmuk/long-video/data/lvu_1.0/lvu_durations.csv').set_index('videoid')

def get_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frames.append(image)
    
    cap.release()
    frames = np.asarray(frames) / 255.0
    return frames

model = timm.create_model('vit_large_patch16_224_in21k', num_classes = 0, pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#check if everything is working fine
x = torch.randn(2, 3, 224, 224).to(device)
y = model.forward_features(x)
print(y.shape)   #this should be of dimention [2,197,1024]

all_ids = []
for csv in glob.glob('/home/csgrad/susimmuk/long-video/data/lvu_1.0/*/*.csv'):
    with open(csv, 'r') as f:
        f.readline()
        for line in f:
            video_id = line.split()[-2].strip()
            all_ids.append(video_id)

print('Total videos: ', len(all_ids))
print('Total unique videos: ', len(set(all_ids)))

random.shuffle(all_ids) #helps if you want to run multiple instances parallelly

cnt = 0
ctr = 0
for video_id in tqdm(all_ids):
    dest = f'{DATA_ROOT}lvu/features/{video_id}.npy' #destination to save features
    if not os.path.exists(dest):
        video_fp = f'{DATA_ROOT}lvu/videos/{video_id}.mp4' #destination of source video
        if not os.path.exists(video_fp):
            ctr+=1
        if os.path.exists(video_fp):
            video = get_video(video_fp)
            video = torch.from_numpy(video.transpose([0, 3, 1, 2])).float()
            duration = duration_data.loc[video_id]['duration']
            # print(cnt, video_id, video.shape, duration)

            features = np.zeros((duration+1, 197, 1024))

            for i in range(int(duration)):
                idx = int(video.shape[0] / duration * i)
                x = torch.unsqueeze(video[idx], 0).to(device)
                x = model.forward_features(x)
                features[i] = x.detach().cpu().numpy()

            x = model.forward_features(torch.unsqueeze(video[-1], 0).to(device))
            features[duration] = x.detach().cpu().numpy()

            np.save(dest, features)
            cnt += 1
print(f"Total videos processed: {cnt}")
print(f"Total videos not found: {ctr}")