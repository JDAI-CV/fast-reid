gpu=2

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501","duke")'  \
OUTPUT_DIR 'logs/beijing/market_duke_softmax_triplet_256_128_bs512'