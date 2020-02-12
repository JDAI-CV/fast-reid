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

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: |
| BagTricks | ImageNet | 93.3% | 85.2% |
| BagTricks + Ibn-a | ImageNet | 94.9% | 87.1% |
| BagTricks + Ibn-a + softMargin | ImageNet | 94.8% | 87.7% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: |
| BagTricks | ImageNet | 86.6% | 77.3% |
| BagTricks + Ibn-a | ImageNet | 88.8% | 78.6% |
| BagTricks + Ibn-a + softMargin | ImageNet | 89.1% | 78.9% |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: |
| BagTricks | ImageNet | 72.0% | 48.6% |
| BagTricks + Ibn-a | ImageNet | 77.7% | 54.6% |
| BagTricks + Ibn-a + softMargin | ImageNet | 77.3% | 55.7% |
