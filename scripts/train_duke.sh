python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("duke",)'  \
DATASETS.TEST_NAMES 'duke' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'cos_triplet' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
OUTPUT_DIR 'logs/2019.9.3'

