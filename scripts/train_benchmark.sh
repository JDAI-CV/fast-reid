GPUS=0,1,2,3

CUDA_VISIBLE_DEVICES=$GPUS python tools/train.py -cfg='configs/resnet_benchmark.yml'
