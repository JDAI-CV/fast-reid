# Hyper-Parameter Optimization in FastReID

This project includes training reid models with hyper-parameter optimization.

Install the following

```bash
pip install 'ray[tune]'
pip install hpbandster ConfigSpace
```

## Training

To train a model with `BOHB`, run

```bash
python3 projects/HPOReID/train_hpo.py --config-file projects/HPOReID/configs/baseline.yml
```

## Known issues
todo