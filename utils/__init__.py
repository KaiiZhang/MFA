# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:36   xin      1.0         None
'''

from .metrics import AvgerageMeter, accuracy, confusion_matrix
from .metrics import evaluate as m_evaluate
from .logging import setup_logger
from .mixup import mixup_data, mixup_criterion, mixup_data_multi_label
from .cut_mix import rand_bbox
from common.optimizer.ranger import Ranger
from common.optimizer.sgd_gc import SGD as SGD_GC

from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np


from common.warmup import WarmupMultiStepLR, GradualWarmupScheduler, WarmUpLR
from common.poly import PolynomialLRDecay, PolynomialLR
from common.CyclicLR import CyclicCosAnnealingLR
from torch.optim.lr_scheduler import StepLR, MultiStepLR

def mfa_lr_scheduler(optimizer, use_warmup=False, iter_num=50000, gamma=0.9, warmup_iter_num=0):
    scheduler = PolynomialLR(optimizer, max_iter=iter_num, gamma=gamma)
    if use_warmup:
        scheduler = WarmUpLR(optimizer, scheduler, warmup_iters=warmup_iter_num)
    return scheduler

def make_optimizer_lr(model, opt, lr, weight_decay, momentum=0.9, nesterov=True):
    base_lr = lr
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        if "layer5" in key:
            lr = base_lr * 10
        params += [{"params": [value], "lr": lr, "initial_lr": lr, "weight_decay": weight_decay}]
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(params, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        # optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
        optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=lr, amsgrad=True)
    elif opt == 'Ranger':
        optimizer = Ranger(params=filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr)
    elif opt == 'SGD_GC':
        optimizer = SGD_GC(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,use_gc=True, gc_conv_only=False)
    else:
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer

def calculate_score(cfg, outputs, targets, metric_type='train'):
    if metric_type=='train':
        f1 = f1_score(
        targets.flatten(),
        outputs.flatten(),
        average="macro")
        acc = accuracy_score(outputs.flatten(), targets.flatten())
        return f1, acc
    else:
        conf_mat = confusion_matrix(pred=outputs.flatten(),
                                            label=targets.flatten(),
                                            num_classes=cfg.MODEL.N_CLASS)
        return conf_mat

class AddDepthChannels:
    def __call__(self, tensor):
        _, h, w = tensor.size()
        for row, const in enumerate(np.linspace(0, 1, h)):
            tensor[1, row, :] = const
        tensor[2] = tensor[0] * tensor[1]
        return tensor

    def __repr__(self):
        return self.__class__.__name__