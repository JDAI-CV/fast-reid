# AGW Baseline in FastReID

Deep Learning for Person Re-identification:  A Survey and Outlook. [arXiv](https://arxiv.org/abs/2001.04193)

This is a re-implementation of [ReID-Survey with a Powerful AGW Baseline](https://github.com/mangye16/ReID-Survey)

## Highlights

- A comprehensive survey with in-depth analysis for person Re-ID in recent years (2016-2019).

- A new evaluation metric, namely mean Inverse Negative Penalty (mINP), which measures the ability to find the hardest correct match.

## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yml>
```

For example, to launch a end-to-end baseline training on market1501 dataset on GPU#1, 
one should excute:

```bash
CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file='configs/AGW_market1501.yml'
```

## Evaluation

To evaluate the model in test set, run similarly:

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <configs.yaml> --eval-only MODEL.WEIGHTS model.pth
```

## Experimental Results

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW |  ImageNet | 94.9% | 87.4% | 63.1% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW |  ImageNet | 89.2% | 79.5% | 44.5% |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW | ImageNet | 76.8% | 53.7% | 12.2% |


```
@article{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={arXiv preprint arXiv:2001.04193},
  year={2020},
}
```
