# FastReID Model Zoo and Baselines

## Introduction

**BoT**:

Bag of Tricks and A Strong Baseline for Deep Person Re-identification. CVPRW2019, Oral.

**AGW**:

This is a re-implementation of [ReID-Survey with a Powerful AGW Baseline](https://github.com/mangye16/ReID-Survey)

**SBS**:

stronger baseline on top of BoT with tricks:
1. Non-local block
2. GeM pooling
3. Circle loss
4. Freeze backbone training
5. Cutout data augmentation & Auto Augmentation
6. Cosine annealing learning rate decay
7. Soft margin triplet loss


### Market1501 Baselines

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 94.4% | 86.1% | 59.4% |
| BoT(R50-ibn) | ImageNet | 94.9% | 87.6% | 64.1% |
| BoT(S50) | ImageNet | 95.1% | 88.5% | 66.0% |
| BoT(R101-ibn) | ImageNet| 95.4% | 88.9% | 67.4% |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 95.3% | 88.2% | 66.3% |
| AGW(R50-ibn) | ImageNet | 95.1% | 88.7% | 67.1% |
| BoT(S50) | ImageNet | - | - | - |
| AGW(R101-ibn) | ImageNet | - | - | - |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 95.4% | 88.2% | 64.8% |
| SBS(R50-ibn) | ImageNet | 95.7% | 89.3% | 67.5% |
| SBS(S50) | ImageNet | 95.0% | 87.0% | 60.6% |
| SBS(R101-ibn) | ImageNet | - | - | - |


### DukeMTMC Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 87.1% | 76.9% | 41.6% |
| BoT(R50-ibn) | ImageNet | 89.6% | 79.1% | 44.4% |
| BoT(S50) | ImageNet | - | - | - |
| BoT(R101-ibn) | ImageNet| - | - |



**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 89.0% | 79.9% | 46.3% |
| AGW(R50-ibn) | ImageNet | 89.8% | 80.7% | 47.7% |
| AGW(R101-ibn) | ImageNet | - | - | - |


**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 89.6% | 79.8% | 44.6% |
| SBS(R50-ibn) | ImageNet | 91.3% | 81.6% | 47.6% |
| SBS(S50) | ImageNet | 90.5% | 79.1% | 42.7% |
| SBS(R101-ibn) | ImageNet | - | - | - |


### MSMT17 Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 72.3%  | 48.3% | 9.7% |
| BoT(R50-ibn) | ImageNet | 77.0% | 54.4% | 12.5% |
| BoT(S50) | ImageNet | 80.4% | 59.2% | 15.9% |
| BoT(R101-ibn) | ImageNet| 79.0% | 57.5% | 14.6% |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 76.7% | 53.6% | 12.2% |
| AGW(R50-ibn) | ImageNet | 79.3% | 57.5% | 14.3% |
| AGW(R101-ibn) | ImageNet | - | - | - |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 83.3% | 59.9% | 14.6% |
| SBS(R50-ibn) | ImageNet | 84.0% | 61.2% | 15.5% |
| SBS(S50) | ImageNet | 82.6% | 58.2% | 13.2% |
| SBS(R101-ibn) | ImageNet | 84.6% | 62.6% | 16.1% |


### VeRi Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50-ibn) | ImageNet | 96.1%  | 78.8% | 43.8% |

### VehicleID Baseline

**BoT**: 
Method: BoT(R50-ibn)

| Testset size | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| Small(800) | ImageNet | 95.5%  | 89.5% | 77.0% |
| Medium(1600) | ImageNet | 93.8%  | 85.6% | 69.4% |
| Large(2400) | ImageNet | 93.3%  | 84.1% | 67.4% |

### VERI-Wild Baseline

**BoT**:
Method: BoT(R50-ibn)

| Testset size | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| Small(3000) | ImageNet | 96.4%  | 87.7% | 69.2% |
| Medium(5000) | ImageNet | 95.1%  | 83.5% | 61.2% |
| Large(10000) | ImageNet | 92.5%  | 77.3% | 49.8% |
