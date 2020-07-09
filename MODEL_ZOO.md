# FastReID Model Zoo and Baselines

## Introduction

This file documents collection of baselines trained with fastreid. All numbers were obtained with 1 NVIDIA P40 GPU.
The software in use were PyTorch 1.4, CUDA 10.1.

In addition to these official baseline models, you can find more models in [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects).

### How to Read the Tables

- The "Name" column contains a link to the config file.
Running `tools/train_net.py` with this config file and 1 GPU will reproduce the model.
- The *model id* column is provided for ease of reference. To check downloaded file integrity, any model on this page contains tis md5 prefix in its file name.
- Training curves and other statistics can be found in `metrics` for each model.

### Common Settings for all Person reid models

**BoT**:

[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf). CVPRW2019, Oral.

**AGW**:

[ReID-Survey with a Powerful AGW Baseline](https://github.com/mangye16/ReID-Survey).

**MGN**:

[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

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

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50.yml) | ImageNet | 94.4% | 86.1% | 59.4% | - |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50-ibn.yml) | ImageNet | 94.9% | 87.6% | 64.1% | - |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_S50.yml) | ImageNet | 95.1% | 88.5% | 66.0% | - |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R101-ibn.yml) | ImageNet| 95.4% | 88.9% | 67.4% | - |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---: |
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50.yml) | ImageNet | 95.3% | 88.2% | 66.3% | - |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50-ibn.yml) | ImageNet | 95.1% | 88.7% | 67.1% | -|
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_S50.yml) | ImageNet | 94.7% | 87.1% | 62.2% | -|
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R101-ibn.yml) | ImageNet | 95.5% | 89.5% | 69.5% | - |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R50.yml) | ImageNet | 95.4% | 88.2% | 64.8% | - |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R50-ibn.yml) | ImageNet | 95.7% | 89.3% | 67.5% | -|
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_S50.yml) | ImageNet | 95.0% | 87.0% | 60.6% | -|
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R101-ibn.yml) | ImageNet | 96.3% | 90.3% | 70.0% | -|

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/mgn_R50-ibn.yml) | ImageNet | 95.8% | 89.7% | 67.0% | -|

### DukeMTMC Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50.yml) | ImageNet | 87.1% | 76.9% | 41.6% | - |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50-ibn.yml) | ImageNet | 89.6% | 79.1% | 44.4% | - |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_S50.yml) | ImageNet | 87.8% | 77.7% | 39.6% | - |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R101-ibn.yml) | ImageNet| 91.1% | 81.3% | 47.7% | -|

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.3% | - |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 89.8% | 80.7% | 47.7% | - |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 89.9% | 79.7% | 44.2% | -| 
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.4% | 82.1% | 50.2% | -|

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R50.yml) | ImageNet | 89.6% | 79.8% | 44.6% | -|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R50-ibn.yml) | ImageNet | 91.3% | 81.6% | 47.6% | -|
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_S50.yml) | ImageNet | 90.5% | 79.1% | 42.7% | -|
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R101-ibn.yml) | ImageNet | 92.4% | 83.2% | 49.7% | -|

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/mgn_R50-ibn.yml) | ImageNet | 91.6% | 82.1% | 46.7% | - |

### MSMT17 Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R50.yml) | ImageNet | 72.3%  | 48.3% | 9.7% | -|
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R50-ibn.yml) | ImageNet | 77.0% | 54.4% | 12.5% | -|
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_S50.yml) | ImageNet | 80.4% | 59.2% | 15.9% | -|
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R101-ibn.yml) | ImageNet| 79.0% | 57.5% | 14.6% | -|

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R50.yml) | ImageNet | 76.7% | 53.6% | 12.2% | -|
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R50-ibn.yml) | ImageNet | 79.3% | 57.5% | 14.3% | -|
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_S50.yml) | ImageNet | 77.3% | 54.7% | 12.6% | -|
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R101-ibn.yml) | ImageNet | 80.8% | 60.2% | 16.5% | -|

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50.yml) | ImageNet | 83.3% | 59.9% | 14.6% | -|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50-ibn.yml) | ImageNet | 84.0% | 61.2% | 15.5% | -|
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_S50.yml) | ImageNet | 82.6% | 58.2% | 13.2% | -|
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R101-ibn.yml) | ImageNet | 85.1% | 63.3% | 16.6% | -|

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/mgn_R50-ibn.yml) | ImageNet | 85.1% | 65.4% | 18.4% | -|

### VeRi Baseline

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:| 
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | ImageNet | 97.0%  | 81.9% | 46.3% | -|

### VehicleID Baseline

**BoT**:  
Test protocol: 10-fold cross-validation; trained on 4 NVIDIA P40 GPU.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center">Method</th>
    <th rowspan="3" align="center">Pretrained</th>
    <th colspan="6" align="center">Testset size</th>
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td colspan="2" align="center">Small</td>
    <td colspan="2" align="center">Medium</td>
    <td colspan="2" align="center">Large</td>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center"><a href="https://github.com/JDAI-CV/fast-reid/blob/master/configs/VehicleID/bagtricks_R50-ibn.yml">BoT(R50-ibn)</a></td>
    <td align="center">ImageNet</td>
    <td align="center">86.6%</td>
    <td align="center">97.9%</td>
    <td align="center">82.9%</td>
    <td align="center">96.0%</td>
    <td align="center">80.6%</td>
    <td align="center">93.9%</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>

### VERI-Wild Baseline

**BoT**:  
Test protocol: Trained on 4 NVIDIA P40 GPU.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center"> Method</th>
    <th rowspan="3" align="center">Pretrained</th>
    <th colspan="9" align="center">Testset size</th>
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td colspan="3" align="center">Small</td>
    <td colspan="3" align="center">Medium</td>
    <td colspan="3" align="center">Large</td>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center"><a href="https://github.com/JDAI-CV/fast-reid/blob/master/configs/VERIWild/bagtricks_R50-ibn.yml">BoT(R50-ibn)</a></td>
    <td align="center">ImageNet</td>
    <td align="center">96.4%</td>
    <td align="center">87.7%</td>
    <td align="center">69.2%</td>
    <td align="center">95.1%</td>
    <td align="center">83.5%</td>
    <td align="center">61.2%</td>
    <td align="center">92.5%</td>
    <td align="center">77.3%</td>
    <td align="center">49.8%</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
