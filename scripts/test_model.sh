gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/test.py -cfg='configs/softmax_triplet.yml' \
INPUT.SIZE_TRAIN '(256, 128)' \
DATASETS.NAMES '("market1501","duke","beijing")'  \
OUTPUT_DIR 'logs/test' \
TEST.WEIGHT 'logs/beijing/market_duke_cuhk03_beijing_revise_bs64/models/model_149.pth'