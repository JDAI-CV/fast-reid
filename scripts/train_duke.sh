gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax.yml' \
DATASETS.NAMES '("duke",)'  \
DATASETS.TEST_NAMES 'duke' \
MODEL.BACKBONE 'resnet50' \
MODEL.IBN 'False' \
INPUT.DO_LIGHTING 'False' \
SOLVER.OPT 'adam' \
OUTPUT_DIR 'logs/2019.8.19/duke/resnet'
