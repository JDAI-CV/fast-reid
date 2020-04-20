gpus='1'

CUDA_VISIBLE_DEVICES=$gpus python train_net.py --config-file 'configs/AGW_dukemtmc.yml'
