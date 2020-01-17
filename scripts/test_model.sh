GPUS=2

CUDA_VISIBLE_DEVICES=$GPUS python tools/test.py -cfg='configs/test_benchmark.yml'
