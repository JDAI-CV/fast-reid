# Stronger Baseline in FastReID

## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yaml>
```

For example, to launch a end-to-end baseline training on market1501 dataset with ibn-net on 4 GPUs, 
one should excute:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --config-file='configs/sbs_market1501.yml'
```

## Experimental Results

stronger baseline tricks: 
1. Non-local block
2. GeM pooling
3. Circle loss 
4. Freeze backbone training 
5. Cutout data augmentation & Auto Augmentation
6. Cosine annealing learning rate decay
7. Soft margin triplet loss

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| stronger baseline(ResNet50-ibn) | ImageNet | 95.5 | 88.4 | 65.8 |
| Robust-ReID | ImageNet | 96.2 | 89.7 | - |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| stronger baseline(ResNet50-ibn) | ImageNet | 91.3 | 81.6 | 47.6 |
| Robust-ReID | ImageNet | 89.8 | 80.3 | - |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| stronger baseline(ResNet50-ibn) | ImageNet | 84.2 | 61.5 | 15.7 |
| ABD-Net | ImageNet | 82.3 | 60.8 | - |
