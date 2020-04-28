gpus='3'

CUDA_VISIBLE_DEVICES=$gpus python train_net.py --config-file 'configs/sbs_msmt17.yml'
