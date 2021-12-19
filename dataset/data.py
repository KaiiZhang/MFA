# -*- encoding: utf-8 -*-
'''
@File    :   data.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 14:11   xin      1.0         None
'''

from torch.utils import data
import os
import torch
import random
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.io import imread
from skimage.color import gray2rgb
import os.path as osp
import numpy as np
import albumentations as A
from PIL import Image
import imageio

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)  #os.walk: traversal all files in rootdir and its subfolders
        for filename in filenames
        if filename.endswith(suffix)
    ]

class BaseDataImageSet(data.Dataset):
    """
    Base dataset for semantic segmentation(train|val).
    Args:
        cfg(yacs.config): Config to the pipeline.
        mode(str): Mode of the pipeline. "train|val".
        img_suffix(str): Suffix of images. ".png|.tif|.jpg".
        seg_map_suffix(str): Suffix of segmentation maps. ".png".
        main_transform(albumentations): Data transform for semantic segmentation.
        reduce_zero_label(bool): Whether to mark label zero as ignored. True|False

    """
    def __init__(self, cfg, mode, dataset_name, img_suffix, seg_map_suffix, main_transform, n_classes, split):
        self.mode = mode
        self.split = split
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.dataset_node = cfg['DATASETS'][self.dataset_name]
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.main_transform = main_transform
        # self.id_to_trainid = eval(self.dataset_node.ID_TO_TRAINID)
        self.n_classes = n_classes

        self.classes_n = self.dataset_node['CLASSES_{}'.format(n_classes)]
        self.valid_classes = self.classes_n['VALID_CLASSES']
        self.classes_names = self.classes_n['CLASSES_NAMES']
        self.void_classes = None if 'void_classes' not in self.classes_n.keys() else self.classes_n['VOID_CLASSES']
        
        if self.split == 'train':
            assert self.dataset_name == 'CITYSCAPES'
            self.file_list = sorted(recursive_glob(rootdir=os.path.join(self.dataset_node.DATA_PATH, 'train'), suffix=self.img_suffix))  #find all files from rootdir and subfolders with suffix = ".png"
        elif self.split == 'val':
            assert self.dataset_name == 'CITYSCAPES'
            self.file_list = sorted(recursive_glob(rootdir=os.path.join(self.dataset_node.DATA_PATH, 'val'), suffix=self.img_suffix))
        elif self.split == 'all':
            self.file_list = sorted(recursive_glob(rootdir=self.dataset_node.DATA_PATH, suffix=self.img_suffix))
        self.num_samples = len(self.file_list)

        self.mean = np.array(self.dataset_node.MEAN)
        self.mean = np.reshape(self.mean, (1,1,3)) 
        self.ignore_index = cfg.DATASETS.IGNORE_INDEX
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        self.resize_trans = self._resize()

        if not self.file_list:
            raise Exception(
                "No files for mode=[%s] found in %s" % (self.mode, self.dataset_node.DATA_PATH,)
            )

    def __getitem__(self, index):
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()
        # read data and main transforms
        img, gt = self.read_data_and_gt(index)

        return img, gt.long(), self.file_list[index]

    def __len__(self):
        return self.num_samples

    def _resize(self):
        # aug = A.Resize(self.cfg.INPUT.SOURCE.SIZE_TRAIN[0], self.cfg.INPUT.SOURCE.SIZE_TRAIN[1], 2)
        aug = A.Resize(self.cfg.INPUT.SOURCE.SIZE_RESIZE[0], self.cfg.INPUT.SOURCE.SIZE_RESIZE[1], 2)
        transform = A.Compose([aug])
        return transform

    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.main_transform.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.main_transform.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index + 1:]

            # replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.main_transform.processors:
                bbox_params = self.main_transform.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.main_transform.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.main_transform.processors:
                keypoint_params = self.main_transform.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.main_transform.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            # recreate transforms
            self.main_transform = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def extract_bbox(self, mask):
        h, w = mask.shape
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0

        return (y1, x1, y2, x2)


    def encode_segmap(self, mask):
        if self.void_classes:
            for _i in self.void_classes:
                mask[mask == _i] = self.ignore_index
        # Put all void classes to zero
        label_copy = self.ignore_index * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy

    def decode_segmap(self, temp):
        r = np.zeros_like(temp)
        g = np.zeros_like(temp)
        b = np.zeros_like(temp)
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[self.to19[l]][0]
            g[temp == l] = self.label_colours[self.to19[l]][1]
            b[temp == l] = self.label_colours[self.to19[l]][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def read_data_and_gt(self, index):
        img_path = self.file_list[index]
        img = imread(img_path).astype(np.float32)
        img = img[...,::-1].copy()
        img -= self.mean # 减去均值

        if self.dataset_name == 'CITYSCAPES':
            if self.split == 'train': # use pseudo label
                gt_path = os.path.join(
                    self.dataset_node.PSEUDO_PATH, 
                    os.path.basename(img_path))
            else:
                gt_path = os.path.join(
                    self.dataset_node.GT_PATH, self.split, 
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_labelIds.png")
        elif self.dataset_name == 'GTA5' or self.dataset_name == 'SYNTHIA':
            gt_path = os.path.join(
                self.dataset_node.GT_PATH, os.path.basename(img_path))
        else:
            raise ValueError('{} are not suporrted!'.format(self.dataset_name))
        if self.dataset_name=='GTA5':
            gt = Image.open(gt_path)
        elif self.dataset_name=='SYNTHIA':
            gt = imageio.imread(gt_path, format='PNG-FI')[:,:,0]
        else:
            gt = imread(gt_path)
        gt = np.array(gt, dtype=np.uint8)
        if self.dataset_name != 'CITYSCAPES' or self.split != 'train':
            # pseudo label do not need encode
            gt = self.encode_segmap(gt)

        data = {'image': img, 'mask': gt}

        if self.dataset_name == 'GTA5' and ( img.shape[0] < self.cfg.INPUT.SOURCE.SIZE_TRAIN[0]  or img.shape[1] < self.cfg.INPUT.SOURCE.SIZE_TRAIN[1] ):
            # some examples' size is smaller than random crop size in GTA5 
            data =  self.resize_trans(**data)
        # base transform
        aug = self.main_transform(**data)
        img, gt = aug['image'], aug['mask']
        return img, gt

    def get_num_samples(self):
        return self.num_samples


class BaseTestDataImageSet(data.Dataset):
    """
        Base dataset for semantic segmentation(test).
        Args:
            cfg(yacs.config): Config to the pipeline.
            img_suffix(str): Suffix of images. ".png|.tif|.jpg".
            main_transform(albumentations): Data transform for semantic segmentation.

        """
    def __init__(self, cfg, img_suffix, main_transform):
        self.cfg = cfg
        self.img_suffix = img_suffix
        self.main_transform = main_transform

        self.file_list = [x for x in os.listdir(cfg.TEST.IMAGE_FOLDER) if x.endswith(img_suffix)]

        self.num_samples = len(self.file_list)

    def read_image(self, img_path, img_type='input', mask=None):
        if img_type == 'input':
            im = imread(img_path) # width, height, nchannel
            mask = im!=im
            mask = np.any(mask, axis=2) # poision of nan
            im[im!=im] = 0 # exchange nan
            return im, mask
        else:
            raise('error')

    def __getitem__(self, index):
        data, filename = self.read_data(index)
        return data, filename

    def __len__(self):
        return self.num_samples

    def read_data(self, index):
       
        img, mask = self.read_image(os.path.join(self.cfg.TEST.IMAGE_FOLDER, self.file_list[index]))
        img = img[:, :, :self.cfg.MODEL.N_CHANNEL]
        data = {'image': img}
        aug = self.main_transform(**data)
        img = aug['image']
        return img, self.file_list[index]

    def get_num_samples(self):
        return self.num_samples


