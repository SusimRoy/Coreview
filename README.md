Coreview: Compact Yet Complete Video Representation
=================================================

This repository contains the official code for the paper as titled above which was publised in IEEE ICIP 2026.

Repository structure
--------------------
- `extract_features/`: feature extraction scripts and integrations.
- `models/`: model architectures and representation modules.
- `runners/`: training and evaluation entrypoints.
- `data/` and `datasets/`: data and dataset loaders.

Datasets
--------
1. Download the Breakfast dataset from [here](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) and place it in `/data`.
2. Download the COIN dataset from [here](https://coin-dataset.github.io/) and place it in `/data`.
3. Download youtube-dl data from the splits provided in data/lvu1.0 data and use yt-dlp. Note that some videos may not be download as they have been made private.

Running on LVU
---------------

-  We used ImageNet21k pretrained ViT dense features from timm. Particularly, we used `vit_large_patch16_224_in21k ViT model`. The feature extraction code:

```
extract_features/extract_features_lvu_vit.py
```

- Run the model on LVU tasks with `main_lvu.py`. Example (used 4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python runners/main_lvu.py
```

Running on Breakfast
--------------------

- We used VideoSwin features (`swin_base_patch244_window877_kinetics600_22k`) for Breakfast. The config file can be found in `extract_features/mmaction2/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py`. Note that the features need to be fixed as Video-Swin-Transformer has removed that support. The feature extraction script is as follows:

```
extract_features/extract_features_breakfast_swin_train.py
extract_features/extract_features_breakfast_swin_test.py
```

- Run the model on Breakfast with `run_breakfast.py`. Example (used 4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python runners/main_breakfast.py
```

Running on COIN
---------------

- Similar to Breakfast, the VideoSwin features (`swin_base_patch244_window877_kinetics600_22k`) for COIN needs to be fixed. Feature extraction scripts:

```
extract_features/extract_features_coin_swin_train.py
extract_features/extract_features_coin_swin_test.py
```

- Run the model on COIN with `run_coin.py`. Example (used 4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python runners/main_coin.py
```


Citation
--------
If you find Coreview useful, please consider citing:
```bibtex
	 @INPROCEEDINGS{11630368,
     author={Roy, Susim and Kaushik, Arjun Ramesh and Ratha, Nalini and Govindaraju, Venu},
     booktitle={2026 IEEE International Conference on Image Processing (ICIP)}, 
     title={Coreview: Compact Yet Complete Video Representation}, 
     year={2026},
     pages={1-6},
     keywords={Videos;Merging;Printing;Accuracy;Computers;Conferences;Modeling;Computer vision;Distance measurement;Equations;Long Video Understanding;Token Compression;Optimal Transport},
     doi={10.1109/ICIP61757.2026.11630368}}
```