gpus='3'

CUDA_VISIBLE_DEVICES=$gpus python3 train_net.py --config-file '/home/zhengkecheng3/zkc/fast-reid/projects/directTransfer/configs/DukeMTMC/sbs_R50.yml'