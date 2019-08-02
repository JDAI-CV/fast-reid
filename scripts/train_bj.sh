gpu=3

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("beijing",)'  \
OUTPUT_DIR 'logs/beijing/market_duke_finetune_bs64'