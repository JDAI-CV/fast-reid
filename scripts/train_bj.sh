gpu=1

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
DATASETS.TEST_NAMES 'bj' \
INPUT.DO_LIGHTING 'False' \
OUTPUT_DIR 'logs/2019.8.14/bj/baseline'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
# DATASETS.TEST_NAMES 'bj' \
# INPUT.DO_LIGHTING 'True' \
# OUTPUT_DIR 'logs/2019.8.9/bj/lighting'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
# DATASETS.TEST_NAMES 'bj' \
# MODEL.BACKBONE 'resnet50_ibn' \
# MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
# INPUT.DO_LIGHTING 'True' \
# OUTPUT_DIR 'logs/2019.8.14/bj/lighting_ibn7_1'