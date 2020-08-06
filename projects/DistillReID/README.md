# Distillation ReID

This project provides a training script of small model
 for both fast inference and high accuracy.


## Datasets Prepration
- Market1501
- DukeMTMC-reID
- MSMT17


## Train and Evaluation
```shell script
# On DukeMTMC-reID dataset
# train BagTricksIBN50 as teacher model
CUDA_VISIBLE_DEVICES=$CUDA python ./tools/train_net.py --config-file ./projects/DistillReID/configs-bagtricks-ibn-dukemtmcreid/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
# train BagTricksIBN18 as student model 
CUDA_VISIBLE_DEVICES=$CUDA python ./projects/DistillReID/train_net.py --kd --config-file ./projects/DistillReID/configs-bagtricks-ibn-dukemtmcreid/KD-bot50ibn-bot18ibn.yml MODEL.DEVICE "cuda:0"
```

## Experimental Reuslts and Pre-trained Model
table class="tg">
<thead>
  <tr>
    <th class="tg-hap0" colspan="2" rowspan="2">Rank-1 (mAP) / Q.Time</th>
    <th class="tg-2oxo" colspan="4">Student (BagTricks)</th>
  </tr>
  <tr>
    <td class="tg-2oxo">IBN-101</td>
    <td class="tg-2oxo">IBN-50</td>
    <td class="tg-2oxo">IBN-34</td>
    <td class="tg-2oxo">IBN-18</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-hap0" rowspan="4">Teacher<br>(BagTricks)</td>
    <td class="tg-2oxo"><span style="font-weight:400;font-style:normal">IBN-101</span></td>
    <td class="tg-2oxo"><span style="font-weight:bold">90.8(80.8)/0.3395s</span></td>
    <td class="tg-2oxo"><span style="font-weight:bold">90.8(81.1)/0.1784s</span></td>
    <td class="tg-2oxo"><span style="font-weight:bold">89.63(78.9)/0.1760s</span></td>
    <td class="tg-2oxo"><span style="font-weight:bold">86.96(75.75)/0.0654s</span></td>
  </tr>
  <tr>
    <td class="tg-2oxo"><span style="font-weight:400;font-style:normal">IBN-50</span></td>
    <td class="tg-2oxo">-</td>
    <td class="tg-2oxo">89.8(79.8)/0.2264s</td>
    <td class="tg-2oxo">88.82(78.9)/0.1761s</td>
    <td class="tg-2oxo">87.75(76.18)/0.0838s</td>
  </tr>
  <tr>
    <td class="tg-2oxo"><span style="font-weight:400;font-style:normal">IBN-34</span></td>
    <td class="tg-2oxo">-</td>
    <td class="tg-2oxo">-</td>
    <td class="tg-2oxo">88.64(76.4)/0.1766s</td>
    <td class="tg-2oxo"></td>
  </tr>
  <tr>
    <td class="tg-2oxo"><span style="font-weight:400;font-style:normal">IBN-18</span></td>
    <td class="tg-2oxo">-</td>
    <td class="tg-2oxo">-</td>
    <td class="tg-2oxo">-</td>
    <td class="tg-2oxo">85.50(71.60)/0.9178s</td>
  </tr>
</tbody>
</table>