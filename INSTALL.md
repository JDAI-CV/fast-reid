# Installation

## Requirements

- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.9
- torchvision that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.

## Installation from PyPI (Recommended)

```bash
pip install fastreid
```

## Installation from Source

```bash
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip install -e .
```

## Optional Dependencies

For GPU support with FAISS:
```bash
pip install fastreid[gpu]
```

For CPU-only FAISS:
```bash
pip install fastreid[faiss]
```

For development:
```bash
pip install fastreid[dev]
```

## Set up with Conda

```bash
conda create -n fastreid python=3.8
conda activate fastreid
conda install pytorch torchvision tensorboard -c pytorch
pip install fastreid
```

## Set up with Docker

Please check the [docker folder](docker)
