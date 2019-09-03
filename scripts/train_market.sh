gpu=2

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("duke",)'  \
DATASETS.TEST_NAMES 'duke' \
INPUT.DO_LIGHTING 'False' \
MODEL.WITH_IBN 'False'  \
MODEL.STAGE_WITH_GCB '(False, False, False, False)' \
SOLVER.LOSSTYPE '("softmax_smooth", "triplet", "center")' \
OUTPUT_DIR 'logs/2019.8.28/duke/smooth_triplet_center'


# MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501",)'  \
# DATASETS.TEST_NAMES 'market1501' \
# SOLVER.IMS_PER_BATCH '64' \
# INPUT.DO_LIGHTING 'True' \
# OUTPUT_DIR 'logs/market/bs64'