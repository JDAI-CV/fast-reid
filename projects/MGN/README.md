# Learning Discriminative Features with Multiple Granularitiesfor Person Re-Identification

Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yml>
```

For example, to launch a end-to-end baseline training on market1501 dataset on GPU#1, 
one should excute:

```bash
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file='configs/mgn_market1501.yml'
```

## Evaluation

To evaluate the model in test set, run similarly:

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <configs.yaml> --eval-only MODEL.WEIGHTS model.pth
```

## Experimental Results

You can reproduce the results by simply excute

```bash
sh scripts/train_market.sh
sh scripts/train_duke.sh
sh scripts/train_msmt.sh
```
### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 95.2% | 88.8% | 63.6% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 89.0% | 80.8% | 44.9% |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 72.2%  | 48.4% | 9.6% |


```
@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM, 
author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}}, 
journal={IEEE Transactions on Multimedia}, 
title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification}, 
year={2019}, 
pages={1-1}, 
doi={10.1109/TMM.2019.2958756}, 
ISSN={1941-0077}, 
}
```
