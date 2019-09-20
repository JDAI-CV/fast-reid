GPUS='2'

CUDA_VISIBLE_DEVICES=$GPUS python tools/test.py -cfg='configs/softmax_triplet.yml' \
DATASETS.TEST_NAMES 'beijing' \
MODEL.NAME 'baseline' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'False' \
TEST.WEIGHT 'logs/beijing/combineall_bs256_cosface/ckpts/model_best.pth'
