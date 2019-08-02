gpu=1

CUDA_VISIBLE_DEVICES=$gpu python tools/test.py -cfg='configs/softmax_triplet.yml' \
INPUT.SIZE_TRAIN '(256, 128)' \
DATASETS.NAMES '("market1501","duke","beijing")'  \
OUTPUT_DIR 'logs/test' \
TEST.WEIGHT 'logs/beijing/market+duke+bj/models/model_149.pth'