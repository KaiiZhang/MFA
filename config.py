'''
Author: Kai Zhang
Date: 2021-11-29 13:27:56
LastEditors: Kai Zhang
LastEditTime: 2021-11-29 23:57:21
Description: creat parameter definiton
'''

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# Alias for easy usage
cfg = _C


_C.MODEL = CN()
_C.MODEL.NAME = "deeplab"
_C.MODEL.N_CHANNEL = 3
_C.MODEL.N_CLASS = 16
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_IDS = "'0'"
_C.MODEL.BACKBONE_NAME = 'resnet101'

_C.MODEL.DROPOUT = 0.0
_C.MODEL.WEIGHT = ''

# _C.MODEL.LOSS_TYPE = 'ce'
# _C.MODEL.CLASS_WEIGHT = []
# _C.MODEL.BC = False
# _C.MODEL.THRESHOLD = 0.5
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 2
_C.DATALOADER.DROP_LAST = False

_C.DATASETS = CN()
_C.DATASETS.IGNORE_INDEX = 255
_C.DATASETS.CLASS_NAMES = ["road","sidewalk","building","wall","fence","pole","traffic_light",
                            "traffic_sign","vegetation","terrain","sky","person","rider","car","truck","bus","train",
                            "motorcycle","bicycle"]
# GTA5
_C.DATASETS.GTA5 = CN()
_C.DATASETS.GTA5.SPLIT='all'
_C.DATASETS.GTA5.DATA_PATH = r''
_C.DATASETS.GTA5.GT_PATH = r''
_C.DATASETS.GTA5.LIST_PATH = r''
_C.DATASETS.GTA5.ID_TO_TRAINID = r'{7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}'
_C.DATASETS.GTA5.MEAN = [104.00698793, 116.66876762, 122.67891434] # BGR RGB:[122.67891434, 116.66876762, 104.00698793]
_C.DATASETS.GTA5.ORIGINALSIZE = [1052, 1914]  # 存在60张分辨率不一致的

_C.DATASETS.GTA5.CLASSES_19 = CN()
_C.DATASETS.GTA5.CLASSES_19.VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
_C.DATASETS.GTA5.CLASSES_19.VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
_C.DATASETS.GTA5.CLASSES_19.CLASSES_NAMES = ["unlabelled","road","sidewalk","building","wall","fence","pole","traffic_light",
                                            "traffic_sign","vegetation","terrain","sky","person","rider","car","truck","bus","train",
                                            "motorcycle","bicycle"]
# CITYSCAPES
_C.DATASETS.CITYSCAPES = CN()
_C.DATASETS.CITYSCAPES.SPLIT='train'
_C.DATASETS.CITYSCAPES.DATA_PATH = r''
_C.DATASETS.CITYSCAPES.GT_PATH = r''
_C.DATASETS.CITYSCAPES.PSEUDO_PATH = r''
_C.DATASETS.CITYSCAPES.LIST_PATH = r''
_C.DATASETS.CITYSCAPES.MEAN = [104.00698793, 116.66876762, 122.67891434]
_C.DATASETS.CITYSCAPES.ORIGINALSIZE = [1024, 2048]
_C.DATASETS.CITYSCAPES.CLASSES_19 = CN()
_C.DATASETS.CITYSCAPES.CLASSES_19.VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
_C.DATASETS.CITYSCAPES.CLASSES_19.CLASSES_NAMES = ["unlabelled","road","sidewalk","building","wall",
                                                    "fence","pole","traffic_light","traffic_sign","vegetation",
                                                    "sky","person","rider","car","bus","motorcycle","bicycle"]
_C.DATASETS.CITYSCAPES.CLASSES_16 = CN()
_C.DATASETS.CITYSCAPES.CLASSES_16.VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33]
_C.DATASETS.CITYSCAPES.CLASSES_16.CLASSES_NAMES = ["unlabelled","road","sidewalk","building","wall",
                                                    "fence","pole","traffic_light","traffic_sign","vegetation",
                                                    "sky","person","rider","car","bus","motorcycle","bicycle"]
_C.DATASETS.CITYSCAPES.CLASSES_13 = CN()
_C.DATASETS.CITYSCAPES.CLASSES_13.VALID_CLASSES = [7, 8, 11, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33]
_C.DATASETS.CITYSCAPES.CLASSES_13.CLASSES_NAMES = ["unlabelled","road","sidewalk","building","traffic_light",
                                                    "traffic_sign","vegetation","sky","person","rider",
                                                    "car","bus","motorcycle","bicycle"]
# SYNTHIA
_C.DATASETS.SYNTHIA = CN()
_C.DATASETS.SYNTHIA.SPLIT='train'
_C.DATASETS.SYNTHIA.DATA_PATH = r''
_C.DATASETS.SYNTHIA.GT_PATH = r''
_C.DATASETS.SYNTHIA.LIST_PATH = r''
_C.DATASETS.SYNTHIA.MEAN = [104.00698793, 116.66876762, 122.67891434]
_C.DATASETS.SYNTHIA.ORIGINALSIZE = [760, 1280]
_C.DATASETS.SYNTHIA.CLASSES_19 = CN()
_C.DATASETS.SYNTHIA.CLASSES_19.VALID_CLASSES = [3,4,2,21,5,7,15,9,6,16,1,10,17,8,18,19,20,12,11]
_C.DATASETS.SYNTHIA.CLASSES_19.CLASSES_NAMES = ["unlabelled","Road","Sidewalk","Building","Wall","Fence","Pole",
                                                "Traffic_light","Traffic_sign","Vegetation","Terrain","sky","Pedestrian",
                                                "Rider","Car","Truck","Bus","Train","Motorcycle","Bicycle",]
_C.DATASETS.SYNTHIA.CLASSES_16 = CN()
_C.DATASETS.SYNTHIA.CLASSES_16.VALID_CLASSES = [3,4,2,21,5,7,15,9,6,1,10,17,8,19,12,11]
_C.DATASETS.SYNTHIA.CLASSES_16.CLASSES_NAMES = ["unlabelled","Road","Sidewalk","Building","Wall",
                                                "Fence","Pole","Traffic_light","Traffic_sign","Vegetation",
                                                "sky","Pedestrian","Rider","Car","Bus","Motorcycle","Bicycle",]
_C.DATASETS.SYNTHIA.CLASSES_13 = CN()
_C.DATASETS.SYNTHIA.CLASSES_13.VALID_CLASSES = [3,4,2,15,9,6,1,10,17,8,19,12,11]
_C.DATASETS.SYNTHIA.CLASSES_13.CLASSES_NAMES = ["unlabelled","Road","Sidewalk","Building","Traffic_light",
                                                "Traffic_sign","Vegetation","sky","Pedestrian","Rider",
                                                "Car","Bus","Motorcycle","Bicycle"]

_C.INPUT = CN()

_C.INPUT.SOURCE = CN()
_C.INPUT.SOURCE.NAME = 'GTA5'
_C.INPUT.SOURCE.IMG_SUFFIX = '.png'
_C.INPUT.SOURCE.SEG_MAP_SUFFIX = '.png'
_C.INPUT.SOURCE.N_CLASSES = 16
_C.INPUT.SOURCE.SPLIT = 'all'
_C.INPUT.SOURCE.SIZE_TRAIN = [1120, 560]
_C.INPUT.SOURCE.SIZE_RESIZE = [512, 512]
_C.INPUT.SOURCE.NORMALIZATION = False
_C.INPUT.SOURCE.USE_RANDOMCROP = False
_C.INPUT.SOURCE.USE_RANDOMRESIZECROP = False
_C.INPUT.SOURCE.USE_RESIZE = False
_C.INPUT.SOURCE.USE_VFLIP = False
_C.INPUT.SOURCE.USE_HFLIP = False
_C.INPUT.SOURCE.USE_RANDOMROTATE90 = False
_C.INPUT.SOURCE.USE_RGBSHIFT = False
_C.INPUT.SOURCE.USE_RANDOMSCALE = False
_C.INPUT.SOURCE.SCALELIMIT = [0.0, 1.0]

_C.INPUT.TARGET = CN()
_C.INPUT.TARGET.NAME = 'CITYSCAPES'
_C.INPUT.TARGET.IMG_SUFFIX = '.png'
_C.INPUT.TARGET.SEG_MAP_SUFFIX = '.png'
_C.INPUT.TARGET.N_CLASSES = 16
_C.INPUT.TARGET.SPLIT = 'train'
_C.INPUT.TARGET.SIZE_TRAIN = [512, 512]
_C.INPUT.TARGET.SIZE_RESIZE = [512, 512]
_C.INPUT.TARGET.SIZE_TEST = [512, 512]
_C.INPUT.TARGET.NORMALIZATION = False
_C.INPUT.TARGET.USE_RANDOMCROP = False
_C.INPUT.TARGET.USE_RANDOMRESIZECROP = False
_C.INPUT.TARGET.USE_RESIZE = False
_C.INPUT.TARGET.USE_VFLIP = False
_C.INPUT.TARGET.USE_HFLIP = False
_C.INPUT.TARGET.USE_RANDOMROTATE90 = False
_C.INPUT.TARGET.USE_RGBSHIFT = False
_C.INPUT.TARGET.USE_RANDOMSCALE = False
_C.INPUT.TARGET.SCALELIMIT = [0.0, 1.0]

_C.INPUT.USE_SOURCE_DATA = False

_C.SOLVER = CN()

# _C.SOLVER.MULTI_SCALE = []
_C.SOLVER.MAX_EPOCHS = 300
_C.SOLVER.MAX_STEPS = 50000
_C.SOLVER.WARMUP_STEP = 2000

_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.MIN_LR = 3e-6
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.LR_SCHEDULER = 'mult_step'
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [40, 70]
_C.SOLVER.T_MAX = 5
_C.SOLVER.MILESTONES = [1]

_C.SOLVER.USE_WARMUP = True
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCH = 10
_C.SOLVER.WARMUP_BEGAIN_LR = 3e-6
_C.SOLVER.WARMUP_METHOD = "linear"

# _C.SOLVER.SWA = False
# _C.SOLVER.SWA_LR = 2e-3
# _C.SOLVER.SWA_START = 75

_C.SOLVER.MIX_PRECISION = False

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.TRAIN_LOG = True
_C.SOLVER.START_SAVE_STEP = 0
_C.SOLVER.START_EVAL_STEP = 0

_C.SOLVER.TENSORBOARD = CN()
_C.SOLVER.TENSORBOARD.USE = True
_C.SOLVER.TENSORBOARD.LOG_PERIOD = 20

_C.SOLVER.PER_BATCH = 4
# _C.SOLVER.FP16 = False
# _C.SOLVER.SYNCBN = False
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_CHECKPOINT_A = r''
_C.SOLVER.RESUME_CHECKPOINT_B = r''
_C.SOLVER.TEMPORAL_CONSIST_WEIGHT = 1.0
_C.SOLVER.CROSS_MODEL_CONSIST_WEIGHT = 0.5
_C.SOLVER.ALPHA_START = 0.2
_C.SOLVER.ALPHA_END = 0.7
_C.SOLVER.RANDOMSEED = 2021


_C.TEST = CN()
_C.TEST.WEIGHT = r'../cache/mfa_result/deeplabv2_A.pkl'
_C.TEST.IMAGE_FOLDER = r''
_C.TEST.FLIP_AUG = False
_C.TEST.ONLY13_CLASSES = False


_C.OUTPUT_DIR = CN()
_C.OUTPUT_DIR = r''
