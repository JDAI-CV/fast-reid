# Model Distillation in FastReID

This project provides a training script of small model
 for both fast inference and high accuracy.


## Datasets Prepration
- Market1501
- DukeMTMC-reID
- MSMT17


## Train and Evaluation
```shell script
# a demo on DukeMTMC-reID dataset
# please see more in ./configs
# train BagTricksIBN50 as teacher model
python3 projects/DistillReID/train_net.py --config-file projects/DistillReID/configs/DukeMTMC/bot50ibn.yml 
# train BagTricksIBN18 as student model 
python3 projects/DistillReID/train_net.py --config-file projects/DistillReID/configs/DukeMTMC/KD-bot50ibn-bot18ibn.yml --kd
```

## Experimental Results and Trained Models

### Settings

All the experiments are conducted with a P40 GPU and 
- CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
- GPU：Tesla P40 (Memory 22919MB)

### DukeMTMC-reID

<table><thead><tr><th colspan="2" rowspan="2">Rank-1 (mAP) / <br>Q.Time/batch(128)</th><th colspan="4">Student (BagTricks)</th></tr><tr><td>IBN-101</td><td>IBN-50</td><td>IBN-34</td><td>IBN-18</td></tr></thead><tbody><tr><td rowspan="4">Teacher<br>(BagTricks)</td><td>IBN-101</td><td>90.8(80.8)/0.3395s</td><td>90.8(81.1)/0.1984s</td><td>89.63(78.9)/0.1760s</td><td>86.96(75.75)/0.0854s</td></tr><tr><td>IBN-50</td><td>-</td><td>89.8(79.8)/0.2264s</td><td>88.82(78.9)/0.1761s</td><td>87.75(76.18)/0.0838s</td></tr><tr><td>IBN-34</td><td>-</td><td>-</td><td>88.64(76.4)/0.1766s</td><td>87.43(75.66)/0.0845s</td></tr><tr><td>IBN-18</td><td>-</td><td>-</td><td>-</td><td>85.50(71.60)/0.9178s</td></tr></tbody></table>

### Market-1501

<table><thead><tr><th colspan="2" rowspan="2">Rank-1 (mAP) / <br>Q.Time/batch(128)</th><th colspan="4">Student (BagTricks)</th></tr><tr><td>IBN-101</td><td>IBN-50</td><td>IBN-34</td><td>IBN-18</td></tr></thead><tbody><tr><td rowspan="4">Teacher<br>(BagTricks)</td><td>IBN-101</td><td>95.43(88.95)/0.2698s</td><td>95.19(89.52)/0.1791s</td><td>94.51(87.82)/0.0869s</td><td>93.85(85.77)/0.0612s</td></tr><tr><td>IBN-50</td><td>-</td><td>95.25(88.16)/0.1823s</td><td>95.13(87.28)/0.0863s</td><td>94.18(85.81)/0.0614s</td></tr><tr><td>IBN-34</td><td></td><td>-</td><td>94.63(84.91)/0.0860s</td><td>93.71(85.20)/0.0620s</td></tr><tr><td>IBN-18</td><td>-</td><td>-</td><td>-</td><td>92.87(81.22)/0.0615s</td></tr><tr><td colspan="2">Average Q.Time</td><td>0.2698s</td><td>0.1807s</td><td>0.0864s</td><td>0.0616s</td></tr></tbody></table>

### MSMT17

<table><thead><tr><th colspan="2" rowspan="2">Rank-1 (mAP) / <br>Q.Time/batch(128)</th><th colspan="4">Student (BagTricks)</th></tr><tr><td>IBN-101</td><td>IBN-50</td><td>IBN-34</td><td>IBN-18</td></tr></thead><tbody><tr><td rowspan="4">Teacher<br>(BagTricks)</td><td>IBN-101</td><td>81.95(60.51)/0.2693s</td><td>82.37(62.08)/0.1792s</td><td>81.07(58.56)/0.0872s</td><td>77.77(52.77)/0.0610s</td></tr><tr><td>IBN-50</td><td>-</td><td>80.18(57.80)/0.1789s</td><td>81.28(58.27)/0.0863s</td><td>78.11(53.10)/0.0623s</td></tr><tr><td>IBN-34</td><td></td><td>-</td><td>78.27(53.41)/0.0873s</td><td>77.65(52.82)/0.0615s</td></tr><tr><td>IBN-18</td><td>-</td><td>-</td><td>-</td><td>74.11(47.26)/0.0621s</td></tr><tr><td colspan="2">Average Q.Time</td><td>0.2693s</td><td>0.1801s</td><td>0.0868s</td><td>0.0617s</td></tr></tbody></table>


## Contact
This project is conducted by [Guan'an Wang](https://wangguanan.github.io/) (guan.wang0706@gmail) and [Xingyu Liao](https://github.com/L1aoXingyu).


