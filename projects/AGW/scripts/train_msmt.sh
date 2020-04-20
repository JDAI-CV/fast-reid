gpus='3'

CUDA_VISIBLE_DEVICES=$gpus python train_net.py --config-file 'configs/AGW_msmt17.yml'
