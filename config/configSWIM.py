# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

cfg = CN()

# Base config files
cfg.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
cfg.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
cfg.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
cfg.DATA.DATA_PATH = ''
# Dataset name
cfg.DATA.DATASET = ''
# Input image size
cfg.DATA.IMG_SIZE = [256,128]
# Interpolation to resize image (random, bilinear, bicubic)
cfg.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
cfg.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
cfg.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
cfg.DATA.PIN_MEMORY = True
# Number of data loading threads
cfg.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
cfg.DATA.MASK_PATCH_SIZE = 24
# [SimMIM] Mask ratio for MaskGenerator
cfg.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
# Model type
cfg.MODEL.TYPE = 'swinv2'
# Model name
cfg.MODEL.NAME = 'swinv2_base_patch4_window16_256'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
cfg.MODEL.PRETRAINED = '/home/gml/HXC/.Apretrain/swinv2_base_patch4_window16_256.pth'
# Checkpoint to resume, could be overwritten by command line argument
cfg.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
cfg.MODEL.NUM_CLASSES = 1000
# Dropout rate
cfg.MODEL.DROP_RATE = 0.2
# Drop path rate
cfg.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
cfg.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
cfg.MODEL.SWIN = CN()
cfg.MODEL.SWIN.PATCH_SIZE = 4
cfg.MODEL.SWIN.IN_CHANS = 3
cfg.MODEL.SWIN.EMBED_DIM = 128
cfg.MODEL.SWIN.DEPTHS = [ 2, 2, 18, 2 ]
cfg.MODEL.SWIN.NUM_HEADS = [ 4, 8, 16, 32 ]
cfg.MODEL.SWIN.WINDOW_SIZE = 8
cfg.MODEL.SWIN.MLP_RATIO = 4.
cfg.MODEL.SWIN.QKV_BIAS = True
cfg.MODEL.SWIN.QK_SCALE = None
#cfg.MODEL.SWIN.APE = False
cfg.MODEL.SWIN.APE = False
cfg.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
cfg.MODEL.SWINV2 = CN()
cfg.MODEL.SWINV2.PATCH_SIZE = 4
cfg.MODEL.SWINV2.IN_CHANS = 3
cfg.MODEL.SWINV2.EMBED_DIM = 128
cfg.MODEL.SWINV2.DEPTHS =  [ 2, 2, 18, 2 ]
cfg.MODEL.SWINV2.NUM_HEADS = [ 4, 8, 16, 32 ]
cfg.MODEL.SWINV2.WINDOW_SIZE = 16
cfg.MODEL.SWINV2.MLP_RATIO = 4.
cfg.MODEL.SWINV2.QKV_BIAS = True
cfg.MODEL.SWINV2.APE = True
cfg.MODEL.SWINV2.PATCH_NORM = True
cfg.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
cfg.TRAIN = CN()
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.EPOCHS = 40
cfg.TRAIN.WARMUP_EPOCHS = 5
cfg.TRAIN.WEIGHT_DECAY = 0.1

cfg.TRAIN.BASE_LR = 1.25e-4
cfg.TRAIN.WARMUP_LR = 1.25e-7
cfg.TRAIN.MIN_LR = 1.25e-6

# cfg.TRAIN.BASE_LR = 1e-4
# cfg.TRAIN.WARMUP_LR = 1e-7
# cfg.TRAIN.MIN_LR = 1e-6

# Clip gradient norm
cfg.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
cfg.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
cfg.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
cfg.TRAIN.USE_CHECKPOINT = False

# LR scheduler
cfg.TRAIN.LR_SCHEDULER = CN()
cfg.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
cfg.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
cfg.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
cfg.TRAIN.LR_SCHEDULER.GAMMA = 0.1
cfg.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
cfg.TRAIN.OPTIMIZER = CN()
cfg.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
cfg.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
cfg.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
cfg.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
cfg.TRAIN.LAYER_DECAY = 1.0

# MoE
cfg.TRAIN.MOE = CN()
# Only save model on master device
cfg.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
cfg.AUG = CN()
# Color jitter factor
cfg.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
cfg.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
cfg.AUG.REPROB = 0.25
# Random erase mode
cfg.AUG.REMODE = 'pixel'
# Random erase count
cfg.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
cfg.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
cfg.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
cfg.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
cfg.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
cfg.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
cfg.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
cfg.TEST = CN()
# Whether to use center crop when testing
cfg.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
cfg.TEST.SEQUENTIAL = False
cfg.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
cfg.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
cfg.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
cfg.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
cfg.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
cfg.TAG = 'default'
# Frequency to save checkpoint
cfg.SAVE_FREQ = 1
# Frequency to logging info
cfg.PRINT_FREQ = 10
# Fixed random seed
cfg.SEED = 0
# Perform evaluation only, overwritten by command line argument
cfg.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
cfg.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
cfg.LOCAL_RANK = 0
# for acceleration
cfg.FUSED_WINDOW_PROCESS = False
cfg.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = cfg.clone()
    #update_config(config, args)

    return config
