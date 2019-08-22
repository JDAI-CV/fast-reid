gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax.yml' \
DATASETS.NAMES '("duke",)'  \
DATASETS.TEST_NAMES 'duke' \
MODEL.BACKBONE 'resnet50' \
MODEL.IBN 'True' \
OUTPUT_DIR 'logs/test'
