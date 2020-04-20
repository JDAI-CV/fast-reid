# Bag of Tricks and A Strong ReID Baseline in FastReID

Bag of Tricks and A Strong Baseline for Deep Person Re-identification. CVPRW2019, Oral.

[Hao Luo\*](https://github.com/michuanhaohao) [Youzhi Gu\*](https://github.com/shaoniangu) [Xingyu Liao\*](https://github.com/L1aoXingyu) [Shenqi Lai](https://github.com/xiaolai-sqlai)

A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification. IEEE Transactions on Multimedia (Accepted).

[[Journal Version(TMM)]](https://ieeexplore.ieee.org/document/8930088)
[[PDF]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
[[Slides]](https://drive.google.com/open?id=1h9SgdJenvfoNp9PTUxPiz5_K5HFCho-V)
[[Poster]](https://drive.google.com/open?id=1izZYAwylBsrldxSMqHCH432P6hnyh1vR)

## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yml>
```

For example, to launch a end-to-end baseline training on market1501 dataset on GPU#1, 
one should excute:

```bash
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file='configs/bagtricks_market1501.yml'
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
| BagTricks | ImageNet | 93.9% | 84.9% | 57.1% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 87.1% | 76.4% | 39.2% |

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