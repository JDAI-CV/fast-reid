gpus='0'

CUDA_VISIBLE_DEVICES=$gpus python train_net.py --config-file 'configs/bagtricks_market1501.yml'
