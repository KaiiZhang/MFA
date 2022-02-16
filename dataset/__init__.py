'''
Author: Kai Zhang
Date: 2021-11-20 18:19:44
LastEditors: Kai Zhang
LastEditTime: 2022-01-07 15:59:14
Description: file content
'''

import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader

from .data import BaseDataImageSet, BaseTestDataImageSet
import albumentations as A
from .copy_paste import CopyPaste
import cv2
from albumentations.pytorch import ToTensorV2
import torch


from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_trm(ipt, is_train=True):
    interpolation=4
    main_transform_list = []
    if is_train:
        if ipt.USE_RESIZE:
            main_transform_list.append(A.Resize(ipt.SIZE_RESIZE[0], ipt.SIZE_RESIZE[1], interpolation=interpolation))
        if ipt.USE_RANDOMSCALE:
            main_transform_list.append(A.RandomScale(always_apply=False, p=1.0, interpolation=interpolation, scale_limit=(ipt.SCALELIMIT[0], ipt.SCALELIMIT[1])))
        if ipt.USE_RANDOMCROP:
            main_transform_list.append(A.RandomCrop(ipt.SIZE_TRAIN[0], ipt.SIZE_TRAIN[1], p=1.0))
        if ipt.USE_VFLIP:
            main_transform_list.append(A.VerticalFlip(p=0.5))
        if ipt.USE_HFLIP:
            main_transform_list.append(A.HorizontalFlip(p=0.5))
        if ipt.USE_RANDOMROTATE90:
            main_transform_list.append(A.RandomRotate90(p=0.5))
        if ipt.USE_RGBSHIFT:
            main_transform_list.append(A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5))
    else:

        if ipt.USE_RESIZE:
            main_transform_list.append(A.Resize(ipt.SIZE_TEST[0], ipt.SIZE_TEST[1], interpolation=interpolation))

    main_transform_list.append(ToTensorV2())
    main_transform = A.Compose(main_transform_list)

    return main_transform


def make_dataloader(cfg, num_gpus):
    source_train_transform = get_trm(cfg.INPUT.SOURCE)
    target_train_transform = get_trm(cfg.INPUT.TARGET)
    target_val_transform = get_trm(cfg.INPUT.TARGET, False)
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus

    source_dataset_name = cfg.INPUT.SOURCE.NAME
    source_train_dataset = BaseDataImageSet(cfg, 'train', source_dataset_name, cfg.INPUT.SOURCE.IMG_SUFFIX,
                                             cfg.INPUT.SOURCE.SEG_MAP_SUFFIX, source_train_transform, cfg.INPUT.SOURCE.N_CLASSES, cfg.INPUT.SOURCE.SPLIT)
    target_dataset_name = cfg.INPUT.TARGET.NAME
    target_train_dataset = BaseDataImageSet(cfg, 'train', target_dataset_name, cfg.INPUT.TARGET.IMG_SUFFIX,
                                             cfg.INPUT.TARGET.SEG_MAP_SUFFIX, target_train_transform, cfg.INPUT.TARGET.N_CLASSES, cfg.INPUT.TARGET.SPLIT)   

    target_val_dataset = BaseDataImageSet(cfg, 'val', target_dataset_name, cfg.INPUT.TARGET.IMG_SUFFIX,
                                             cfg.INPUT.TARGET.SEG_MAP_SUFFIX, target_val_transform, cfg.INPUT.TARGET.N_CLASSES, 'val')
    source_train_loader = DataLoaderX(
        source_train_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=True,
        num_workers=num_workers, pin_memory=True,   drop_last=cfg.DATALOADER.DROP_LAST)

    target_train_loader = DataLoaderX(
        target_train_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=True,
        num_workers=num_workers, pin_memory=True,   drop_last=cfg.DATALOADER.DROP_LAST)

    target_val_loader = DataLoaderX(
        target_val_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers)
    
    return source_train_loader, target_train_loader, target_val_loader




def make_ddp_dataloader(cfg, world_size, rank):
    source_train_transform = get_trm(cfg.INPUT.SOURCE)
    target_train_transform = get_trm(cfg.INPUT.TARGET)
    target_val_transform = get_trm(cfg.INPUT.TARGET, False)
    num_workers = cfg.DATALOADER.NUM_WORKERS 

    source_dataset_name = cfg.INPUT.SOURCE.NAME
    source_train_dataset = BaseDataImageSet(cfg, 'train', source_dataset_name, cfg.INPUT.SOURCE.IMG_SUFFIX,
                                             cfg.INPUT.SOURCE.SEG_MAP_SUFFIX, source_train_transform, cfg.INPUT.SOURCE.N_CLASSES, cfg.INPUT.SOURCE.SPLIT)

    target_dataset_name = cfg.INPUT.TARGET.NAME
    target_train_dataset = BaseDataImageSet(cfg, 'train', target_dataset_name, cfg.INPUT.TARGET.IMG_SUFFIX,
                                             cfg.INPUT.TARGET.SEG_MAP_SUFFIX, target_train_transform, cfg.INPUT.TARGET.N_CLASSES, cfg.INPUT.TARGET.SPLIT)   
    target_val_dataset = BaseDataImageSet(cfg, 'val', target_dataset_name, cfg.INPUT.TARGET.IMG_SUFFIX,
                                             cfg.INPUT.TARGET.SEG_MAP_SUFFIX, target_val_transform, cfg.INPUT.TARGET.N_CLASSES, 'val')

    source_train_sampler = torch.utils.data.distributed.DistributedSampler(source_train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    target_train_sampler = torch.utils.data.distributed.DistributedSampler(target_train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    target_val_sampler = torch.utils.data.distributed.DistributedSampler(target_val_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)

    source_train_loader = torch.utils.data.DataLoader(dataset=source_train_dataset,
                                               batch_size=cfg.SOLVER.PER_BATCH,
                                               shuffle=False,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=True,
                                               sampler=source_train_sampler)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_train_dataset,
                                               batch_size=cfg.SOLVER.PER_BATCH,
                                               shuffle=False,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=True,
                                               sampler=target_train_sampler)
    target_val_loader = torch.utils.data.DataLoader(dataset=target_val_dataset,
                                               batch_size=cfg.SOLVER.PER_BATCH,
                                               shuffle=False,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=True,
                                               sampler=target_val_sampler)

    return source_train_loader, target_train_loader, target_val_loader, source_train_sampler, target_train_sampler, target_val_sampler





def make_dataset(cfg):
    source_train_transform = get_trm(cfg.INPUT.SOURCE)
    target_train_transform = get_trm(cfg.INPUT.TARGET)
    target_val_transform = get_trm(cfg.INPUT.TARGET, False)

    source_dataset_name = cfg.INPUT.SOURCE.NAME
    source_train_dataset = BaseDataImageSet(cfg, 'train', source_dataset_name, cfg.INPUT.SOURCE.IMG_SUFFIX,
                                             cfg.INPUT.SOURCE.SEG_MAP_SUFFIX, source_train_transform, cfg.INPUT.SOURCE.N_CLASSES, cfg.INPUT.SOURCE.SPLIT)
    target_dataset_name = cfg.INPUT.TARGET.NAME
    target_train_dataset = BaseDataImageSet(cfg, 'train', target_dataset_name, cfg.INPUT.TARGET.IMG_SUFFIX,
                                             cfg.INPUT.TARGET.SEG_MAP_SUFFIX, target_train_transform, cfg.INPUT.TARGET.N_CLASSES, cfg.INPUT.TARGET.SPLIT)   

    target_val_dataset = BaseDataImageSet(cfg, 'val', target_dataset_name, cfg.INPUT.TARGET.IMG_SUFFIX,
                                             cfg.INPUT.TARGET.SEG_MAP_SUFFIX, target_val_transform, cfg.INPUT.TARGET.N_CLASSES, 'val')
    return source_train_dataset, target_train_dataset, target_val_dataset

def make_val_dataloader(target_val_dataset, cfg):
    target_val_loader = DataLoaderX(
        target_val_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS)
    return target_val_loader
    

if __name__ == '__main__':
    from config import cfg
    import argparse
    config_file = '/home/lvyifan/zkai/uda/MFA_Final/configs/mfa.yml'
    cfg.merge_from_file(config_file)
    source_train_loader, target_train_loader, target_val_loader = make_dataloader(cfg, 0)
    # print(len(source_train_loader))
    # print(len(target_train_loader))
    # print(len(target_val_loader))