gpu=0,1

CUDA_VISIBLE_DEVICES=$gpu python tools/train.py -cfg='configs/resnet_ibn_bj.yml'
