GPUS=2

CUDA_VISIBLE_DEVICES=$GPUS python tools/train.py -cfg='configs/veri_tricks.yml'