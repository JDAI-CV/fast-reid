gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/test.py -cfg='configs/softmax_triplet.yml' \
MODEL.BACKBONE 'resnet50' \
INPUT.SIZE_TRAIN '(256, 128)' \
DATASETS.NAMES '("market1501","duke","beijing")'  \
DATASETS.TEST_NAMES 'market1501' \
OUTPUT_DIR 'logs/test' \
TEST.WEIGHT 'logs/market/bs64_light/models/model_149.pth'