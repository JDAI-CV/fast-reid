gpus='2'

CUDA_VISIBLE_DEVICES=$gpus python3 train_net.py --eval-only --config-file '/home/zhengkecheng3/zkc/fast-reid/projects/directTransfer/configs/Market1501/bagtricks_R50.yml'