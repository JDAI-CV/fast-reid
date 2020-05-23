# FastReID Model Zoo and Baselines

## Introduction

**BoT**:

Bag of Tricks and A Strong Baseline for Deep Person Re-identification. CVPRW2019, Oral.

**AGW**:

This is a re-implementation of [ReID-Survey with a Powerful AGW Baseline](https://github.com/mangye16/ReID-Survey)

**MGN**:

**SBS**:

stronger baseline on top of BoT:

Bag of Freebies(BoF):

1. Circle loss
2. Freeze backbone training
3. Cutout data augmentation & Auto Augmentation
4. Cosine annealing learning rate decay
5. Soft margin triplet loss

Bag of Specials(BoS):

1. Non-local block
2. GeM pooling

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
| AGW(S50) | ImageNet | 94.7% | 87.1% | 62.2% |
| AGW(R101-ibn) | ImageNet | 95.5% | 89.5% | 69.5% |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 95.4% | 88.2% | 64.8% |
| SBS(R50-ibn) | ImageNet | 95.7% | 89.3% | 67.5% |
| SBS(S50) | ImageNet | 95.0% | 87.0% | 60.6% |
| SBS(R101-ibn) | ImageNet | 96.3% | 90.3% | 70.0% |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50-ibn) | ImageNet | 95.8% | 89.7% | 67.0% |

### DukeMTMC Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 87.1% | 76.9% | 41.6% |
| BoT(R50-ibn) | ImageNet | 89.6% | 79.1% | 44.4% |
| BoT(S50) | ImageNet | 87.8% | 77.7% | 39.6% |
| BoT(R101-ibn) | ImageNet| 91.1% | 81.3% | 47.7% |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 89.0% | 79.9% | 46.3% |
| AGW(R50-ibn) | ImageNet | 89.8% | 80.7% | 47.7% |
| AGW(S50) | ImageNet | 89.9% | 79.7% | 44.2% |
| AGW(R101-ibn) | ImageNet | 91.4% | 82.1% | 50.2% |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 89.6% | 79.8% | 44.6% |
| SBS(R50-ibn) | ImageNet | 91.3% | 81.6% | 47.6% |
| SBS(S50) | ImageNet | 90.5% | 79.1% | 42.7% |
| SBS(R101-ibn) | ImageNet | 92.4% | 83.2% | 49.7% |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50-ibn) | ImageNet | 91.6% | 82.1% | 46.7% |

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
| AGW(S50) | ImageNet | 77.3% | 54.7% | 12.6% |
| AGW(R101-ibn) | ImageNet | 80.8% | 60.2% | 16.5% |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 83.3% | 59.9% | 14.6% |
| SBS(R50-ibn) | ImageNet | 84.0% | 61.2% | 15.5% |
| SBS(S50) | ImageNet | 82.6% | 58.2% | 13.2% |
| SBS(R101-ibn) | ImageNet | 85.1% | 63.3% | 16.6% |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | 82.9% | 61.2% | 14.9% |
| SBS(R50-ibn) | ImageNet | 85.1% | 65.4% | 18.4% |
| SBS(R101-ibn) | ImageNet | 96.3% | 90.3% | 70.0% |

### VeRi Baseline

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50-ibn) | ImageNet | 97.0%  | 81.3% | 44.9% |

### VehicleID Baseline

**BoT**: 
Method: BoT(R50-ibn+gem pooling+weighted triplet+soft margin)

| Testset size | Pretrained | Rank@1 | Rank@5 |
| :---: | :---: | :---: |:---: |
| Small(800) | ImageNet | 88.0%  | 98.2% |
| Medium(1600) | ImageNet | 83.0%  | 96.2% |
| Large(2400) | ImageNet | 80.7%  | 94.4% |

### VERI-Wild Baseline

**BoT**:
Method: BoT(R50-ibn+gem pooling+weighted triplet+soft margin)

| Testset size | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| Small(3000) | ImageNet | 96.4%  | 87.7% | 69.2% |
| Medium(5000) | ImageNet | 95.1%  | 83.5% | 61.2% |
| Large(10000) | ImageNet | 92.5%  | 77.3% | 49.8% |
