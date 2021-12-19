'''
Author: your name
Date: 2021-11-09 20:50:29
LastEditTime: 2021-11-29 20:26:05
LastEditors: Kai Zhang
Description: In User Settings Edit
FilePath: /zkai/uda/MFA/model/deeplab_v2.py
'''
import torch
from torch import nn
import torch.nn.functional as F


from .modules.utils import initialize_weights
from .adaptation_model import Deeplab

Norm2d = nn.BatchNorm2d
class DeepV2(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """
    def __init__(self, in_channels, n_classes, backbone, model_path, droprate=0.0):
        super(DeepV2, self).__init__()


        base_model = Deeplab(num_classes=n_classes, droprate=droprate)

        if model_path:
            base_model.load_param(model_path)

        # input_dim >3
        if in_channels > 3:
            with torch.no_grad():
                pretrained_conv1 = base_model.conv1.weight.clone()
                base_model.conv1 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
                torch.nn.init.kaiming_normal_(
                    base_model.conv1.weight, mode='fan_out', nonlinearity='relu')
                # Re-assign pretraiend weights to first 3 channels
                # (assuming alpha channel is last in your input data)
                base_model.conv1.weight[:, :3] = pretrained_conv1
        
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.layer5 = base_model.layer5

        initialize_weights(self.layer5)

    def forward(self, x, upsample=True):
        x_size = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        s2_features = x
        x = self.layer2(x)
        s4_features = x
        x = self.layer3(x)
        x = self.layer4(x)
        pred = self.layer5(x)
        if upsample:
            pred = F.interpolate(pred, x_size[2:], mode='bilinear', align_corners=True)
        return pred