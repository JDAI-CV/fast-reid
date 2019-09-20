gpu=2

#CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
#DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
#DATASETS.TEST_NAMES 'bj' \
#MODEL.BACKBONE 'resnet50' \
#MODEL.WITH_IBN 'False' \
#MODEL.STAGE_WITH_GCB '(False, False, False, False)' \
#SOLVER.LOSSTYPE '("softmax_smooth", "triplet")' \
#OUTPUT_DIR 'logs/2019.8.26/bj/softmax_smooth'


CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
DATASETS.TEST_NAMES 'bj' \
INPUT.DO_LIGHTING 'False' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
 MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
MODEL.STAGE_WITH_GCB '(False, False, False, False)' \
SOLVER.LOSSTYPE '("softmax_smooth", "triplet")' \
OUTPUT_DIR 'logs/2019.8.27/bj/ibn_softmax_smooth'

# CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
# DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
# DATASETS.TEST_NAMES 'bj' \
# MODEL.BACKBONE 'resnet50_ibn' \
# INPUT.DO_LIGHTING 'True' \
# OUTPUT_DIR 'logs/2019.8.14/bj/lighting_ibn7_1'