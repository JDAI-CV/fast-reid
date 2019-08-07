gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501","duke","cuhk03","beijing")'  \
SOLVER.IMS_PER_BATCH '64' \
OUTPUT_DIR 'logs/beijing/market_duke_cuhk03_beijing_revise_bs64_light'