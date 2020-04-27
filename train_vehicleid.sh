gpus='0,1,2,3'
CUDA_VISIBLE_DEVICES=$gpus python train_net.py --config-file 'configs/bagtricks_vehicleid.yml'