# FastReID Developer Guide

This guide is for developers who want to understand FastReID's architecture and extend its functionality.

## Project Architecture

FastReID follows a modular design with clear separation of concerns:

```
fastreid/
├── config/          # Configuration management using yacs
├── data/            # Data loading, processing, and augmentation
│   ├── datasets/    # Dataset definitions
│   ├── samplers/    # Batch sampling strategies
│   └── transforms/  # Image transformations
├── engine/          # Training and evaluation loops
├── evaluation/      # ReID-specific evaluation metrics
├── modeling/        # Neural network architectures
│   ├── backbones/   # Feature extraction networks
│   ├── heads/       # Classification/embedding heads
│   └── losses/      # Loss function implementations
├── solver/          # Optimizers and learning rate schedulers
├── tools/           # Command-line interfaces
└── utils/           # Helper functions and utilities
```

## Adding New Components

FastReID uses a registry-based system for extensibility. Here's how to add new components:

### Adding a New Dataset

1. **Create a dataset file** in `fastreid/data/datasets/`:

```python
from .bases import ReidDataset
from ..data_utils import read_image_paths
from fastreid.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MyNewDataset(ReidDataset):
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        # Load your data here
        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        
        super().__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path):
        # Parse filenames and extract person IDs, camera IDs
        return data
```

2. **Register and use** in config files:
```yaml
DATASETS:
  NAMES: ["MyNewDataset"]
```

### Adding a New Backbone

1. **Create a backbone file** in `fastreid/modeling/backbones/`:

```python
import torch.nn as nn
from fastreid.utils.registry import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
def build_my_backbone(cfg):
    # Build your backbone architecture
    model = MyBackboneNetwork()
    return model
```

2. **Use in config files**:
```yaml
MODEL:
  BACKBONE:
    NAME: "build_my_backbone"
```

## Configuration System

FastReID uses YACS for configuration management. Key principles:

- **Hierarchical configs**: Organize settings in logical groups
- **Type safety**: YACS enforces type checking
- **Defaults**: Always provide sensible defaults
- **Documentation**: Comment your config options

## Testing Guidelines

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Model tests**: Verify model loading and inference
- **Performance tests**: Benchmark critical paths

## Performance Optimization

- **Profiling**: Use PyTorch profiler to identify bottlenecks
- **Memory**: Monitor GPU memory usage
- **Data loading**: Optimize data pipeline performance
- **Mixed precision**: Use automatic mixed precision when possible

## Best Practices

- **Code organization**: Keep related functionality together
- **Error handling**: Provide informative error messages
- **Documentation**: Document public APIs thoroughly
- **Backward compatibility**: Maintain compatibility when possible
- **Performance**: Profile before optimizing

For more information, see the main [README](README.md) and [documentation](https://fast-reid.readthedocs.io/).
