'''
Author: Kai Zhang
Date: 2021-11-29 21:19:05
LastEditors: Kai Zhang
LastEditTime: 2021-11-30 15:54:05
Description: file content
'''
import torch
if __name__ == '__main__':
    # model_path = '/home/lvyifan/zkai/uda/cache/mfa_syn_result/deeplabv2_A.pkl'
    # save_path = '/home/lvyifan/zkai/uda/cache/mfa_syn_result/deeplabv2_A.pkl'
    model_path = '/home/lvyifan/zkai/uda/cache/from_synthia_to_cityscapes_on_deeplab_best_emma_model.pkl'
    save_path = '/home/lvyifan/zkai/uda/cache/from_synthia_to_cityscapes_on_deeplab_best_emma_model.pkl'
    param_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['ResNet101']['model_state']
    torch.save(param_dict, save_path)