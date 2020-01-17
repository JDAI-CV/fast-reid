from yacs.config import CfgNode as CN

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
_C.MODEL.NAME = 'baseline'
_C.MODEL.DIST_BACKEND = 'dp'
# Model backbone
_C.MODEL.BACKBONE = 'resnet50'
# Last stride for backbone
_C.MODEL.LAST_STRIDE = 1
# If use IBN block
_C.MODEL.WITH_IBN = False
# If use SE block
_C.MODEL.WITH_SE = False
# Global Context Block configuration
_C.MODEL.STAGE_WITH_GCB = (False, False, False, False)
_C.MODEL.GCB = CN()
_C.MODEL.GCB.ratio = 1./16.
# Model head
_C.MODEL.HEAD = 'softmax'
# If use imagenet pretrain model
_C.MODEL.PRETRAIN = True
# Pretrain model path
_C.MODEL.PRETRAIN_PATH = ''
# Checkpoint for continuing training
_C.MODEL.CHECKPOINT = ''
_C.MODEL.VERSION = ''

#
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# If use mask
_C.INPUT.USE_MASK = False
# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
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
_C.INPUT.RE.DO = True
_C.INPUT.RE.PROB = 0.5
_C.INPUT.RE.MEAN = [0.340*255, 0.326*255, 0.316*255]
# Cutout
_C.INPUT.CUTOUT = CN()
_C.INPUT.CUTOUT.DO = False
_C.INPUT.CUTOUT.PROB = 0.5
_C.INPUT.CUTOUT.SIZE = 64
_C.INPUT.CUTOUT.MEAN = [0, 0, 0]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("market1501",)
# List of the dataset names for testing
_C.DATASETS.TEST_NAMES = "market1501"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.DIST = False

_C.SOLVER.OPT = "adam"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3
_C.SOLVER.CE_SMOOTH_ON = False

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.LOG_INTERVAL = 30
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.NORM = True
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"
