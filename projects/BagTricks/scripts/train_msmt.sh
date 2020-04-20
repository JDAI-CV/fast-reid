gpus='2'

CUDA_VISIBLE_DEVICES=$gpus python train_net.py --config-file 'configs/bagtricks_msmt17.yml'
