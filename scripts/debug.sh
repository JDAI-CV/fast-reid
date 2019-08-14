gpu=2

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501",)'  \
DATASETS.TEST_NAMES 'market1501' \
INPUT.DO_LIGHTING 'True' \
INPUT.MIXUP 'True' \
OUTPUT_DIR 'logs/test'
