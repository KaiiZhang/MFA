'''
Author: Kai Zhang
Date: 2021-11-09 20:50:29
LastEditors: Kai Zhang
LastEditTime: 2021-11-30 16:25:34
Description: start testing shell
'''

import os

#  GTA5-to-CITYSCAPES
os.system('python test.py --config_file configs/mfa.yml -g 3')

#  SYNTHIA-to-CITYSCAPES
os.system('python test.py --config_file configs/mfa_syn.yml -g 3')