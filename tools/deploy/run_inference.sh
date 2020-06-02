
python caffe_inference.py --model-def "logs/caffe_R34/baseline_R34.prototxt" \
--model-weights "logs/caffe_R34/baseline_R34.caffemodel" \
--input \
'/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1182_c5s3_015240_04.jpg' \
'/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1182_c6s3_038217_01.jpg' \
'/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1183_c5s3_006943_05.jpg' \
--output "caffe_R34_output"