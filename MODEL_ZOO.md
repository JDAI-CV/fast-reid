# FastReID Model Zoo and Baselines

## Introduction


### Market1501 Baselines

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 94.1% | 85.9% | 59.3% |
| BoT(R50-ibn) | ImageNet | - | - | - |
| BoT(S50) | ImageNet | - | - | - |
| BoT(R101-ibn) | ImageNet| - | - |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 94.9% | 87.4% | 63.1% |
| AGW(R50-ibn) | ImageNet | - | - | - |
| AGW(R101-ibn) | ImageNet | - | - | - |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | - | - | - |
| SBS(R50-ibn) | ImageNet | 95.5% | 88.7% | 66.4% |
| SBS(S50) | ImageNet | - | - | - |
| SBS(R101-ibn) | ImageNet | - | - | - |


### DukeMTMC Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 86.1% | 75.9% | 38.7% |
| BoT(R50-ibn) | ImageNet | 89.0% | 78.8% | 43.6% |
| BoT(S50) | ImageNet | - | - | - |
| BoT(R101-ibn) | ImageNet| - | - |



**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 88.9% | 79.1% | 43.2% |
| AGW(R50-ibn) | ImageNet | - | - | - |
| AGW(R101-ibn) | ImageNet | - | - | - |


**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | - | - | - |
| SBS(R50-ibn) | ImageNet | 91.3% | 81.6% | 47.6% |
| SBS(S50) | ImageNet | - | - | - |
| SBS(R101-ibn) | ImageNet | - | - | - |


### MSMT17 Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 70.4%  | 47.5% | 9.6% |
| BoT(R50-ibn) | ImageNet | 76.9% | 55.0% | 13.5% |
| BoT(S50) | ImageNet | - | - | - |
| BoT(R101-ibn) | ImageNet| - | - |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| AGW(R50) | ImageNet | 75.6% | 52.6% | 11.9% |
| AGW(R50-ibn) | ImageNet | - | - | - |
| AGW(R101-ibn) | ImageNet | - | - | - |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| SBS(R50) | ImageNet | - | - | - |
| SBS(R50-ibn) | ImageNet | 84.2% | 61.5% | 15.7% |
| SBS(S50) | ImageNet | - | - | - |
| SBS(R101-ibn) | ImageNet | - | - | - |


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
