# Person Attribute Recognition in FastReID

## Training and Evaluation

To train a model, run:

```bash
python3 projects/PartialReID/train_net.py --config-file <config.yaml> --num-gpus 1
```

For example, to train the attribute recognition network with ResNet-50 Backbone in PA100k dataset,
one should execute:

```bash
python3 projects/attribute_recognition/train_net.py --config-file projects/attribute_recognition/configs/pa100.yml --num-gpus 4
```

## Results

### PA100k

| Method | mA | Accu | Prec | Recall | F1 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Strongbaseline | 77.76 | 77.59 | 88.38 | 84.35 | 86.32 |

More datasets and test results are waiting to add, stay tune!
