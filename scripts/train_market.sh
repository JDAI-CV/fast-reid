gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501",)'  \
DATASETS.TEST_NAMES 'market1501' \
MODEL.BACKBONE 'resnet50_ibn' \
MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
INPUT.DO_LIGHTING 'True' \
OUTPUT_DIR 'logs/2019.8.14/market/lighting_ibn_7_1'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# MODEL.BACKBONE 'resnet50_ibn' \
# MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
# INPUT.DO_LIGHTING 'False' \
# OUTPUT_DIR 'logs/2019.8.13/market/ibn7_1'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# SOLVER.IMS_PER_BATCH '64' \
# INPUT.DO_LIGHTING 'True' \
# OUTPUT_DIR 'logs/market/bs64'