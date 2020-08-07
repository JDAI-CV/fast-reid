# Distillation ReID

This project provides a training script of small model
 for both fast inference and high accuracy.


## Datasets Prepration
- Market1501
- DukeMTMC-reID
- MSMT17


## Train and Evaluation
```shell script
# a demo on DukeMTMC-reID dataset
# train BagTricksIBN50 as teacher model
CUDA_VISIBLE_DEVICES=$CUDA python ./tools/train_net.py --config-file ./projects/DistillReID/configs-bagtricks-ibn-dukemtmcreid/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
# train BagTricksIBN18 as student model 
CUDA_VISIBLE_DEVICES=$CUDA python ./projects/DistillReID/train_net.py --kd --config-file ./projects/DistillReID/configs-bagtricks-ibn-dukemtmcreid/KD-bot50ibn-bot18ibn.yml MODEL.DEVICE "cuda:0"
```

## Experimental Results and Trained Models

### DukeMTMC-reID

<table><thead><tr><th colspan="2" rowspan="2">Rank-1 (mAP) / <br>Q.Time/batch(128)</th><th colspan="4">Student (BagTricks)</th></tr><tr><td>IBN-101</td><td>IBN-50</td><td>IBN-34</td><td>IBN-18</td></tr></thead><tbody><tr><td rowspan="4">Teacher<br>(BagTricks)</td><td>IBN-101</td><td>90.8(80.8)/0.3395s</td><td>90.8(81.1)/0.1984s</td><td>89.63(78.9)/0.1760s</td><td>86.96(75.75)/0.0854s</td></tr><tr><td>IBN-50</td><td>-</td><td>89.8(79.8)/0.2264s</td><td>88.82(78.9)/0.1761s</td><td>87.75(76.18)/0.0838s</td></tr><tr><td>IBN-34</td><td>-</td><td>-</td><td>88.64(76.4)/0.1766s</td><td></td></tr><tr><td>IBN-18</td><td>-</td><td>-</td><td>-</td><td>85.50(71.60)/0.9178s</td></tr></tbody></table>

### Market-1501

<table><thead><tr><th colspan="2" rowspan="2">Rank-1 (mAP) / <br>Q.Time/batch(128)</th><th colspan="4">Student (BagTricks)</th></tr><tr><td>IBN-101</td><td>IBN-50</td><td>IBN-34</td><td>IBN-18</td></tr></thead><tbody><tr><td rowspan="4">Teacher<br>(BagTricks)</td><td>IBN-101</td><td>95.43(88.95)/0.2698s</td><td>95.19(89.52)/0.1791s</td><td>94.51(87.82)/0.0869s</td><td>93.85(85.77)/0.0612s</td></tr><tr><td>IBN-50</td><td>-</td><td>95.25(88.16)/0.1823s</td><td>95.13(87.28)/0.0863s</td><td>93.44(82.92)/0.0621s</td></tr><tr><td>IBN-34</td><td></td><td>-</td><td>94.63(84.91)/0.0860s</td><td></td></tr><tr><td>IBN-18</td><td>-</td><td>-</td><td>-</td><td>92.87(81.22)/0.0615s</td></tr><tr><td colspan="2">Average Q.Time</td><td>0.2698s</td><td>0.1807s</td><td>0.0864s</td><td>0.0616s</td></tr></tbody></table>

### MSMT17

Comming Soon


## Contact
This project is conducted by Guan'an Wang (guan.wang0706@gmail) and Liao Xingyu.