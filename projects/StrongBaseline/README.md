# Strong Baseline in FastReID

## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yaml>
```

For example, to launch a end-to-end baseline training on market1501 dataset with ibn-net on 4 GPUs, 
one should excute:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --config-file='configs/baseline_ibn_market1501.yml'
```

## Experimental Results

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 93.6% | 85.1% | 58.1% |
| BagTricks + Ibn-a | ImageNet | 94.8% | 87.3% | 63.5% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 86.1% | 75.9% | 38.7% |
| BagTricks + Ibn-a | ImageNet | 89.0% | 78.8% | 43.6% |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 70.4%  | 47.5% | 9.6% |
| BagTricks + Ibn-a | ImageNet | 76.9% | 55.0% | 13.5% |
