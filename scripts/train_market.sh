#!/usr/bin/env bash
gpu=0

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# MODEL.BACKBONE 'resnet50' \
# MODEL.IBN 'False' \
# OUTPUT_DIR 'logs/2019.8.20/market/resnet_softmax'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# MODEL.BACKBONE 'resnet50' \
# MODEL.IBN 'False' \
# OUTPUT_DIR 'logs/2019.8.20/market/resnet_softmax_triplet'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# MODEL.BACKBONE 'resnet50' \
# MODEL.IBN 'True' \
# MODEL.PRETRAIN_PATH '/home/user01/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
# OUTPUT_DIR 'logs/2019.8.20/market/resnet_ibn_softmax'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# MODEL.BACKBONE 'resnet50' \
# MODEL.IBN 'True' \
# MODEL.PRETRAIN_PATH '/home/user01/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
# OUTPUT_DIR 'logs/2019.8.20/market/resnet_ibn_softmax_triplet'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax.yml' \
# DATASETS.NAMES '("duke",)'  \
# DATASETS.TEST_NAMES 'duke' \
# MODEL.BACKBONE 'resnet50' \
# MODEL.IBN 'False' \
# OUTPUT_DIR 'logs/2019.8.20/duke/resnet_softmax'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("duke",)'  \
# DATASETS.TEST_NAMES 'duke' \
# MODEL.BACKBONE 'resnet50' \
# MODEL.IBN 'False' \
# OUTPUT_DIR 'logs/2019.8.20/duke/resnet_softmax_triplet'

 CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
 DATASETS.NAMES '("market1501",)'  \
 DATASETS.TEST_NAMES 'market1501' \
 MODEL.BACKBONE 'resnet50' \
 MODEL.WITH_IBN 'True' \
 MODEL.PRETRAIN_PATH '/home/user01/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
 MODEL.STAGE_WITH_GCB '(False, False, False, False)' \
 SOLVER.LOSSTYPE '("softmax_smooth", "triplet")' \
 OUTPUT_DIR 'logs/2019.8.25/market/ibn_smooth'

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501",)'  \
DATASETS.TEST_NAMES 'market1501' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
MODEL.PRETRAIN_PATH '/home/user01/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
MODEL.STAGE_WITH_GCB '(False, True, True, True)' \
SOLVER.LOSSTYPE '("softmax_smooth", "triplet")' \
OUTPUT_DIR 'logs/2019.8.25/market/ibn_gc_smooth'

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("duke",)'  \
DATASETS.TEST_NAMES 'duke' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
MODEL.PRETRAIN_PATH '/home/user01/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
MODEL.STAGE_WITH_GCB '(False, False, False, False)' \
SOLVER.LOSSTYPE '("softmax_smooth", "triplet")' \
OUTPUT_DIR 'logs/2019.8.25/duke/ibn_smooth'
