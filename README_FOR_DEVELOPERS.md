# newFastReID: Developer and Contributor Guide:

This guide is for those who want to take a look into the newFastReID codebase, extend its functionality, or contribute back to the project.

This version has been tested on **Python 3.11** and is compatible with **Python 3.8+**.

---

## 1. Setting Up a Development Environment:

To contribute to newFastReID, install it from source in an editable mode.

### Steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://https://github.com/WhiteMetagross/newFastReID.git
    cd newFastReID
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv newFastReID
    source venv/bin/activate  #On Windows, use `venv\Scripts\activate`
    ```

3.  **Install in Editable Mode:**
    This command links the installed package to your source files, so any changes made are immediately reflected.
    ```bash
    pip install -e .
    ```

4.  **Install Development Dependencies:**
    Install `pre-commit` to ensure your contributions adhere to the project's code style.
    ```bash
    pip install pre-commit
    pre-commit install
    ```

---

## 2. Project Architecture:

The project's folder and file structure (see [newFastReID CodeBase Index](MODEL_ZOO_COMPATIBILITY.md) for further details):

```text
newFastReID/
├── config/      #Manages the configuration system using yacs. `defaults.py` holds the master config.
├── data/        #Handles all data loading, processing, and augmentation.
│   ├── datasets/  #Contains definitions for each supported dataset.
│   ├── samplers/  #Implements batch sampling strategies (like triplet sampling).
│   └── transforms/ #Contains image transformation and augmentation logic.
├── engine/      #Contains the core training and evaluation loops.
│   ├── defaults.py #Defines the `DefaultTrainer`.
│   └── hooks.py    #A modular system for adding functionality to the training loop (like logging, checkpointing).
├── evaluation/  #Implements ReID specific evaluation metrics like mAP and CMC.
├── modeling/    #Defines the neural network architectures.
│   ├── backbones/ #Foundational networks like ResNet, OSNet.
│   ├── heads/     #The final layers that produce the feature embeddings.
│   └── losses/    #Implementation of various loss functions (like Triplet Loss, Cross Entropy).
├── solver/      #Contains optimizers and learning rate schedulers.
├── tools/       #The entry points for the command line scripts (fastreid-train, etc.).
└── utils/       #A collection of helper functions for logging, checkpointing, and distributed training.
```

---

## 3. Extension of the newFastReID:

newFastReID is designed to be modular. Adding new components usually involves creating a new file and registering your component.

### Example: Adding a New Dataset

1.  **Create a Dataset File:**
    Add a new Python file in `newFastReID/data/datasets/`, for example, `mynewdataset.py`.

2.  **Implement the Dataset Class:**
    Your class should inherit from `ReidDataset` and handle loading image paths, person IDs (pids), and camera IDs (camids).

    ```python
    from .bases import ReidDataset
    from ..data_utils import read_image_paths
    from fastreid.utils.registry import DATASET_REGISTRY

    @DATASET_REGISTRY.register()
    class MyNewDataset(ReidDataset):
        def __init__(self, root='datasets', **kwargs):
            self.root = root
            # ... your logic to load data ...
            train = self._process_dir(self.train_dir)
            query = self._process_dir(self.query_dir)
            gallery = self._process_dir(self.gallery_dir)
            # ...
            super().__init__(train, query, gallery, **kwargs)

        def _process_dir(self, dir_path):
            # ... your logic to parse file names and extract pids/camids ...
            return data
    ```

3.  **Register the custom Dataset:**
    The `@DATASET_REGISTRY.register()` decorator makes your dataset available to the framework.

4.  **Use it in a Config File:**
    You can now use `"MyNewDataset"` in your YAML configuration file under `DATASETS.NAMES`.

### Example: Adding a New Backbone

1.  **Create a Backbone File:**
    Add a new file in `fastreid/modeling/backbones/`, like `mynewbackbone.py`.

2.  **Implement and Register the Backbone:**
    Define your model and register it with the `BACKBONE_REGISTRY`.

    ```python
    import torch.nn as nn
    from fastreid.utils.registry import BACKBONE_REGISTRY

    @BACKBONE_REGISTRY.register()
    def build_mynew_backbone(cfg):
        # ... logic to construct your backbone ...
        model = nn.Sequential(...)
        return model
    ```

3.  **Use it in a Config File:**
    Set `MODEL.BACKBONE.NAME` to `"build_mynew_backbone"` in your YAML config.

---

## 4. Running Tests:

To maintain code quality and prevent regressions, it's crucial to run the test suite.

```bash
python -m unittest discover tests
```

Before submitting a pull request, please ensure all existing tests pass and, if you're adding new functionality, include new tests to cover it.

## 5. Contribution Guidelines:

1.  **Fork the repository** and create your branch from `main`.
2.  **Follow the code style.** The `pre-commit` hooks will help enforce this automatically.
3.  **Make clear, concise commits.**
4.  **Write tests** for any new functionality.
5.  **Update documentation** if you are changing user-facing APIs.
6.  **Submit a pull request** with a detailed description of your changes.

---

## License:

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
