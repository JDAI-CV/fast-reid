# AGW Baseline in FastReID

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
| AGW |  ImageNet | 95.2% | 87.9% |
| AGW + Ibn-a | ImageNet | 95.1% | 88.2% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: |
| AGW |  ImageNet | 88.4% | 79.4% |
| AGW + Ibn-a | ImageNet | 89.3% | 80.2% |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: |
| AGW | ImageNet | | |
| AGW + Ibn-a | ImageNet | |
