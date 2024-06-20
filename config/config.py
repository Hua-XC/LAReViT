from yacs.config import CfgNode as CN
cfg = CN()

cfg.SEED = 0

# dataset
cfg.DATASET = 'sysu'    # sysu or regdb
cfg.DATA_PATH_SYSU = '/home/gml/HXC/dataset/SYSU-MM01/'
cfg.DATA_PATH_RegDB = '/home/gml/HXC/RegDB/'
cfg.PRETRAIN_PATH = '/home/gml/HXC/.Apretrain/A/jx_vit_base_p16_224-80ecf9dd.pth'

cfg.START_EPOCH = 1
cfg.MAX_EPOCH = 32

cfg.H = 288
cfg.W = 144
cfg.BATCH_SIZE = 24  # num of images for each modality in a mini batch
cfg.NUM_POS = 4

# PMT
cfg.METHOD ='PMT'
cfg.PL_EPOCH = 0    # for PL strategy
cfg.MSEL = 0.5      # weight for MSEL
cfg.DCL = 0.5       # weight for DCL
cfg.MARGIN = 0.1    # margin for triplet


# model
cfg.STRIDE_SIZE =  [10,10]
cfg.DROP_OUT = 0.03
cfg.ATT_DROP_RATE = 0.0
cfg.DROP_PATH = 0.1

# optimizer
cfg.OPTIMIZER_NAME = 'AdamW'  # AdamW or SGD
cfg.MOMENTUM = 0.9    # for SGD
#ori
cfg.BASE_LR = 3e-4
#cfg.BASE_LR = 1e-3
cfg.WEIGHT_DECAY = 1e-4
cfg.WEIGHT_DECAY_BIAS = 1e-4
cfg.BIAS_LR_FACTOR = 1

cfg.LR_PRETRAIN = 0.5
cfg.LR_MIN = 0.01
cfg.LR_INIT = 0.01
cfg.WARMUP_EPOCHS = 3








