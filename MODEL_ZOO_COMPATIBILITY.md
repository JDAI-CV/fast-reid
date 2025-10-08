# FastReID Model Zoo Compatibility Guide:

This document describes the improved FastReID library's compatibility with models from the [FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md).

## What's Fixed:

The FastReID library has been enhanced to work with **all kinds of FastReID models** from the model zoo, including:

- **SBS (Shake Both Sides)** models (like `market_sbs_R101-ibn.pth`).
- **BOT (Bag of Tricks)** models (like `veriwild_bot_R50-ibn.pth`).
- **AGW (Attention Guided Weighting)** models.
- **MGN (Multiple Granularities Network)** models.
- **Baseline** models.
- Different ResNet depths (18, 34, 50, 101, 152).
- IBN (Instance Batch Normalization) variants.
- SE (Squeeze and Excitation) variants.
- Non Local variants.

## Usuage:

### 1. Generate Configuration for Any Model

```bash
#For any FastReID model, generate a compatible configuration:
python generate_config.py --model "path/to/your/model.pth" --output model_config.yml

#This creates two files:
# - model_config.yml (with metadata).
# - model_config_clean.yml (ready to use).
```

### 2. Use with Demo Script:

```bash
#Extract features from images:
python -m fastreid.tools.demo \
    --config-file model_config_clean.yml \
    --input path/to/images/*.jpg \
    --output features_output
```

### 3. Use in the Code:

```python
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

# Load configuration:
cfg = get_cfg()
cfg.merge_from_file('model_config_clean.yml')
cfg.MODEL.DEVICE = 'cuda'  # or 'cpu'
cfg.freeze()

#Build and load model:
model = build_model(cfg)
model.eval()

#The model is ready for inference.
```

## Key Improvements Made:

### 1. Enhanced Architecture Detection:
- **Fixed ResNet depth detection**: Now correctly identifies ResNet 101 models (was showing as ResNet 34)
- **Improved meta architecture detection**: Detects SBS, AGW, MGN, BOT from filenames when not in layer names
- **Better backbone analysis**: Uses layer3 block count for accurate depth detection

### 2. Fixed Configuration Generation:
- **YAML serialization**: Fixed tuple serialization issues that caused parsing errors.
- **Metadata handling**: Automatically creates clean configs without metadata for direct use.
- **Validation improvements**: Better error handling and temporary file cleanup.

### 3. Enhanced Model Loading:
- **Checkpoint format compatibility**: Handles different checkpoint formats (`model`, `state_dict`, or direct).
- **Architecture mapping**: Maps unsupported architectures (SBS, AGW, BOT) to Baseline for compatibility.
- **Pixel normalization**: Extracts and preserves model-specific normalization parameters.

### 4. Demo Script Fixes:
- **Image processing**: Fixed negative stride issues in BGR to RGB conversion.
- **Error handling**: Better error messages and graceful failure handling.

## Tested Models:

The following models have been verified to work:

| Model Type | Architecture | Backbone | Status |
|------------|-------------|----------|---------|
| VeRiWild BOT | Baseline | ResNet50-IBN | Working |
| Market1501 SBS | Baseline | ResNet101-IBN | Working |
| DukeMTMC AGW | Baseline | ResNet50-IBN | Compatible |
| MSMT17 MGN | MGN | ResNet50-IBN | Compatible |

## Advanced Usage:

### Custom Model Weights Path:

Specify model weights in the config file:

```yaml
MODEL:
  META_ARCHITECTURE: Baseline
  WEIGHTS: "/path/to/your/model.pth"
  # ... rest of config
```

### Batch Processing:

```python
#Process multiple images
import torch
import cv2

images = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 256))  #width, height
    img = img[:, :, ::-1].copy()  #BGR to RGB
    img = img.transpose(2, 0, 1)  #HWC to CHW
    images.append(img)

batch = torch.from_numpy(np.stack(images)).float()
with torch.no_grad():
    features = model({"images": batch})
```

## Testing Setup:

Run the compatibility test to verify everything works:

```bash
python test_model_compatibility.py
```

This will test both working and previously problematic models to ensure compatibility.

## Notes:

- **Architecture Mapping**: SBS, AGW, and BOT models are mapped to the `Baseline` architecture since they're typically training techniques rather than different model architectures
- **Feature Dimensions**: All models output 2048 dimensional features by default.
- **Input Size**: Standard input size is 256x128 (height x width).
- **Device Support**: Works on both CPU and GPU.

## Troubleshooting:

### Config Loading Errors:
- Use the `_clean.yml` version of generated configs.
- Remove any `_META_` sections from config files.

### Model Loading Errors:
- Ensure the model path is correct and accessible.
- Check that the model file is not corrupted.
- Verify the model is from the FastReID model zoo.

### Demo Script Issues:
- Ensure input images exist and are readable.
- Check that the output directory is writable.
- Use absolute paths when possible.

## Future Enhancements:

- Support for more backbone architectures (Vision Transformers, etc.).
- Automatic dataset detection and configuration.
- Integration with more evaluation metrics.
- Support for multi scale testing.

---

For more information, see the [FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md).
