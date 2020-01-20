GPUS=1

CUDA_VISIBLE_DEVICES=$GPUS python tools/train_net.py -cfg='configs/baseline.yml'