from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'Baseline'

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = 50
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "StandardHead"
_C.MODEL.HEADS.POOL_LAYER = 'avgpool'
_C.MODEL.HEADS.NUM_CLASSES = 751

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.SMOOTH_ON = False
_C.MODEL.LOSSES.EPSILON = 0.1
_C.MODEL.LOSSES.SCALE_CE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.MARGIN = 0.3
_C.MODEL.LOSSES.NORM_FEAT = False
_C.MODEL.LOSSES.SCALE_TRI = 1.0

# Path (possibly with schema like catalog:// or detectron2://) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]
#

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random lightning and contrast change 
_C.INPUT.DO_LIGHTING = False
_C.INPUT.BRIGHTNESS = 0.4
_C.INPUT.CONTRAST = 0.4
# Random erasing
_C.INPUT.RE = CN()
_C.INPUT.RE.ENABLED = False
_C.INPUT.RE.PROB = 0.5
_C.INPUT.RE.MEAN = [0.596*255, 0.558*255, 0.497*255]
# Cutout
_C.INPUT.CUTOUT = CN()
_C.INPUT.CUTOUT.ENABLED = False
_C.INPUT.CUTOUT.PROB = 0.5
_C.INPUT.CUTOUT.SIZE = 64
_C.INPUT.CUTOUT.MEAN = [0.485*255, 0.456*255, 0.406*255]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.SCHED = "warmup"
# warmup config
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

# cosine annealing
_C.SOLVER.DELAY_ITERS = 100
_C.SOLVER.COS_ANNEAL_ITERS = 100

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.LOG_PERIOD = 30
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.EVAL_PERIOD = 50
_C.TEST.IMS_PER_BATCH = 128

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

