gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
MODEL.BACKBONE 'resnet50_ibn' \
MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
DATASETS.NAMES '("market1501",)'  \
DATASETS.TEST_NAMES 'market1501' \
SOLVER.IMS_PER_BATCH '64' \
INPUT.DO_LIGHTING 'False' \
OUTPUT_DIR 'logs/market/resnet50_ibn_bs64_256x128'