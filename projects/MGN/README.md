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
@article{Wang2018LearningDF,
  title={Learning Discriminative Features with Multiple Granularities for Person Re-Identification},
  author={Guanshuo Wang and Yufeng Yuan and Xiong Chen and Jiwei Li and Xi Zhou},
  journal={Proceedings of the 26th ACM international conference on Multimedia},
  year={2018}
}
```
