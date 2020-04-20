# AGW Baseline in FastReID


## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yaml>
```

For example, to launch a end-to-end baseline training on market1501 dataset with ibn-net on 4 GPUs, 
one should excute:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --config-file='configs/AGW_market1501.yml'
```

## Experimental Results

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW |  ImageNet | 94.9% | 87.4% | 63.1% |

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW |  ImageNet | 88.9% | 79.1% | 43.2% |

### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW | ImageNet | 75.6% | 52.6% | 11.9% |
