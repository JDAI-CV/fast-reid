gpus='0'
CUDA_VISIBLDE_DEVICES=$gpus python demo.py --config-file 'logs/market1501/baseline/config.yaml' \
--input \
'/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1182_c5s3_015240_04.jpg' \
'/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1182_c6s3_038217_01.jpg' \
'/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1183_c5s3_006943_05.jpg' \
--output 'logs/market1501/baseline/' \
--opts MODEL.WEIGHTS 'logs/market1510/baseline/model_final.pth'

