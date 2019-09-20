GPUS='2,3'

CUDA_VISIBLE_DEVICES=$GPUS python tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501","dukemtmc","cuhk03","msmt17")'  \
DATASETS.TEST_NAMES 'beijing' \
SOLVER.IMS_PER_BATCH '256' \
MODEL.NAME 'mgn_plus' \
MODEL.WITH_IBN 'False' \
MODEL.BACKBONE 'resnet50' \
MODEL.VERSION 'combineall_bs256_mgn_plus' \
SOLVER.OPT 'adam' \
SOLVER.LOSSTYPE '("softmax", "triplet")'



# MODEL.PRETRAIN_PATH '/home/liaoxingyu2/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
