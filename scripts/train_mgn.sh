GPUS='2'

CUDA_VISIBLE_DEVICES=$GPUS python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("dukemtmc",)'  \
DATASETS.TEST_NAMES 'dukemtmc' \
INPUT.SIZE_TRAIN '[288, 144]' \
INPUT.SIZE_TEST '[288, 144]' \
SOLVER.IMS_PER_BATCH '64' \
MODEL.NAME 'mgn' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'mgn++' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
