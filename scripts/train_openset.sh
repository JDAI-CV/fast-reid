GPUS='0,1'

CUDA_VISIBLE_DEVICES=$GPUS python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("dukemtmc",)'  \
DATASETS.TEST_NAMES 'dukemtmc' \
INPUT.SIZE_TRAIN '[256, 128]' \
INPUT.SIZE_TEST '[256, 128]' \
SOLVER.IMS_PER_BATCH '256' \
MODEL.NAME 'baseline' \
MODEL.WITH_IBN 'True' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'baseline_bs256' \
MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")' \
