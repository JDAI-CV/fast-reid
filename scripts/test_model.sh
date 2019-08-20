gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/test.py -cfg='configs/softmax_triplet.yml' \
MODEL.BACKBONE 'resnet50' \
MODEL.IBN 'True' \
MODEL.PRETRAIN 'False' \
DATASETS.TEST_NAMES 'duke' \
OUTPUT_DIR 'logs/test' \
TEST.WEIGHT 'logs/2019.8.16/market/resnet50_ibn_1_1/models/model_149.pth'