'''
Author: Kai Zhang
Date: 2021-11-09 20:50:29
LastEditors: Kai Zhang
LastEditTime: 2021-11-30 16:49:23
Description: start training shell
'''

import os

#  GTA5-to-CITYSCAPES
os.system('python train.py --config_file configs/mfa.yml -g 2 -p 12999')

#  SYNTHIA-to-CITYSCAPES
os.system('python train.py --config_file configs/mfa_syn.yml -g 2 -p 12999')