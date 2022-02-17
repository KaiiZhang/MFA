'''
Author: Kai Zhang
Date: 2021-11-29 13:27:56
LastEditors: Please set LastEditors
LastEditTime: 2022-02-17 15:41:21
Description: train file
'''

import os
import torch
from torch import nn 
import argparse

from config import cfg
from utils import setup_logger
from dataset import make_ddp_dataloader
from model import build_model
from trainer import BaseTrainer
from torch.cuda.amp import autocast as autocast, GradScaler
import random
import torch.multiprocessing as mp
import torch.distributed as dist
from utils import make_optimizer_lr
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

def cleanup():
    dist.destroy_process_group()

def train(gpu, args, cfg):
    logger = setup_logger('baseline', cfg.OUTPUT_DIR, 0)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(2021)
    model_A = build_model(cfg)
    model_B = build_model(cfg)
    torch.cuda.set_device(gpu)
    model_A.cuda(gpu)
    model_B.cuda(gpu)
    
    model_A = nn.SyncBatchNorm.convert_sync_batchnorm(model_A)
    model_B = nn.SyncBatchNorm.convert_sync_batchnorm(model_B)

    optimizer_A = make_optimizer_lr(model_A, opt=cfg.SOLVER.OPTIMIZER_NAME, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)
    optimizer_B = make_optimizer_lr(model_B, opt=cfg.SOLVER.OPTIMIZER_NAME, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)

    scaler = GradScaler(enabled=cfg.SOLVER.MIX_PRECISION)
    
    model_A = nn.parallel.DistributedDataParallel(model_A, device_ids=[gpu])
    model_B = nn.parallel.DistributedDataParallel(model_B, device_ids=[gpu])

    source_train_loader, target_train_loader, target_val_dataloader, source_train_sampler, target_train_sampler, target_val_sampler = make_ddp_dataloader(cfg, args.world_size, rank)
    
    trainer = BaseTrainer(cfg, args, gpu, model_A, model_B, source_train_loader, target_train_loader, target_val_dataloader, optimizer_A, optimizer_B, scaler)
    if rank == 0:
        logger.info(type(model_A))
        logger.info(trainer)
    for e in range(cfg.SOLVER.MAX_EPOCHS):

        source_train_sampler.set_epoch(e)
        target_train_sampler.set_epoch(e)

        for batch in (tqdm(trainer.tt_dl) if rank==0 else trainer.tt_dl):
            trainer.step(batch)
            trainer.handle_new_batch()
            if trainer.current_iteration>=cfg.SOLVER.MAX_STEPS:
                break
        if trainer.current_iteration>=cfg.SOLVER.MAX_STEPS:
                break
        trainer.handle_new_epoch()
    if rank == 0:
        torch.optim.swa_utils.update_bn(trainer.tt_dl, trainer.mean_model_A)
        torch.optim.swa_utils.update_bn(trainer.tt_dl, trainer.mean_model_B)
        trainer.mean_save()
    trainer.evaluate(trainer.mean_model_A)
    trainer.evaluate(trainer.mean_model_B)
    cleanup()





def main():
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("--config_file", default="./configs/mfa.yml", help="path to config file", type=str)
    parser.add_argument("-n", "--nodes", default=1, help="number of node", type=int, metavar='N')
    parser.add_argument("-g", "--gpus", default=1, help="number of gpu", type=int)
    parser.add_argument("-nr", "--nr", default=0, help="rank within the nodes", type=int)
    parser.add_argument("-p", "--port", default='12355', help="master port", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_gpus = args.gpus
    logger = setup_logger('baseline', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(cfg))

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    # cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * cfg.SOLVER.PER_BATCH * args.world_size / 4  # LR are linear to batch size
    cfg.freeze()

    mp.spawn(train, nprocs=args.gpus, args=(args, cfg))
if __name__ == "__main__":
    main()
