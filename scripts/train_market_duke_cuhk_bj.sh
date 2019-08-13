gpu=1

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
# DATASETS.TEST_NAMES 'bj' \
# OUTPUT_DIR 'logs/2019.8.9/bj/baseline'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
# DATASETS.TEST_NAMES 'bj' \
# INPUT.DO_LIGHTING 'True' \
# OUTPUT_DIR 'logs/2019.8.9/bj/lighting'

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
MODEL.BACKBONE 'resnet50_ibn' \
MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
DATASETS.TEST_NAMES 'bj' \
INPUT.DO_LIGHTING 'True' \
OUTPUT_DIR 'logs/2019.8.12/bj/ibn_lighting'