'''
Author: Kai Zhang
Date: 2021-11-09 20:50:29
LastEditors: Kai Zhang
LastEditTime: 2021-11-29 19:41:35
Description: file content
'''

from .unet import Unet
from .res_unet import Res_Unet

from .efficient_unet import EfficientUnet

from .hr_seg import HRNet
from .deeplab_v3_plus import DeepV3Plus
from .deeplab_v2 import DeepV2
from .adaptation_model import Deeplab
from .dlinknet import DLinkNet
from .msc import MSC
from .msc_hierarchical import Basic
# from .double_res_unet import Double_Res_Unet


def build_model(cfg):
    if cfg.MODEL.NAME == "deeplabv2":
        model = DeepV2(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.BACKBONE_NAME, cfg.MODEL.WEIGHT, cfg.MODEL.DROPOUT)
    else:
        raise KeyError('Not supoort model name: ', cfg.MODEL.NAME)

    return model

