'''
Author: Kai Zhang
Date: 2021-11-29 13:27:56
LastEditors: Kai Zhang
LastEditTime: 2022-02-23 15:09:58
Description: creat trainer for train.py
'''

from utils import AvgerageMeter, calculate_score, m_evaluate, mfa_lr_scheduler

import logging
import os
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
from torch.cuda.amp import autocast as autocast
import torch.distributed as dist
class result_recorder(object):
    def __init__(self, name, best_res=0.0, best_iter=0):
        self.name = name
        self.best_res = best_res
        self.best_iter = best_iter
    
    def update(self, res, cur_iter):
        if res>self.best_res:
            self.best_res = res
            self.best_iter = cur_iter

    def __str__(self):
        return 'The best result of {} is {:.4f} in {} iters \n'.format(self.name, self.best_res, self.best_iter)
class BaseTrainer(object):
    def __init__(self, cfg, args, gpu,  model_A, model_B, source_train_loader, target_train_loader,
                 target_val_loader,  optimizer_A, optimizer_B, scaler):

        self.cfg = cfg
        self.args = args
        self.gpu = gpu
        self.rank = args.nr * args.gpus + gpu
        self.model_A = model_A
        self.model_B = model_B

        self.st_dl = source_train_loader
        self.tt_dl = target_train_loader
        self.tv_dl = target_val_loader

        if cfg.INPUT.USE_SOURCE_DATA:
            self.st_iter = iter(self.st_dl)

        self.loss_avg = AvgerageMeter()
        self.acc_avg = AvgerageMeter()
        self.f1_avg = AvgerageMeter()

        self.val_loss_avg = AvgerageMeter()
        self.val_acc_avg = AvgerageMeter()

        self.res_recoder_A = result_recorder('mean_model_A')
        self.res_recoder_B = result_recorder('mean_model_B')

        self.train_epoch = 1
        self.optim_A = optimizer_A
        self.optim_B = optimizer_B
        self.scaler = scaler
        if cfg.SOLVER.RESUME:
            self.load_param(self.model_A, cfg.SOLVER.RESUME_CHECKPOINT_A)
            self.load_param(self.model_B, cfg.SOLVER.RESUME_CHECKPOINT_B)

        self.batch_cnt = 0
        self.logger = logging.getLogger('baseline.train')
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        self.epochs = cfg.SOLVER.MAX_EPOCHS
        if self.rank==0 and cfg.SOLVER.TENSORBOARD.USE:
            summary_dir = os.path.join(cfg.OUTPUT_DIR, 'summaries/')
            os.makedirs(summary_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=summary_dir)
        self.current_iteration = 0

        self.mean_model_A = torch.optim.swa_utils.AveragedModel(self.model_A, device=gpu)
        self.mean_model_B = torch.optim.swa_utils.AveragedModel(self.model_B, device=gpu)
        self.mean_model_A.update_parameters(self.model_A)
        self.mean_model_B.update_parameters(self.model_B)

        assert self.is_equal(self.model_A, self.mean_model_A.module)
        assert self.is_equal(self.model_B, self.mean_model_B.module)
        
        self.scheduler_A = mfa_lr_scheduler(self.optim_A, self.cfg.SOLVER.USE_WARMUP, self.cfg.SOLVER.MAX_STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_STEP )
        self.scheduler_B = mfa_lr_scheduler(self.optim_B, self.cfg.SOLVER.USE_WARMUP, self.cfg.SOLVER.MAX_STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_STEP )

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255).to(self.gpu)
        self.consist_loss = nn.MSELoss().to(self.gpu)
        self.temporal_consist_weight = torch.tensor(cfg.SOLVER.TEMPORAL_CONSIST_WEIGHT).to(gpu)
        self.cross_model_consist_weight = torch.tensor(cfg.SOLVER.CROSS_MODEL_CONSIST_WEIGHT).to(gpu)
        self.adjust_valid_alpha()

    def is_equal(self, model, mean_model):
        unequal_key = []
        for key in mean_model.state_dict().keys():
            if torch.any(model.state_dict()[key]!=mean_model.state_dict()[key]):
                unequal_key.append(key)
        return len(unequal_key)==0

    def load_param(self, model, weight_path):
        if self.rank == 0:
            print("Resume from checkpoint {}".format(weight_path))
        param_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        if 'state_dict' in param_dict.keys():
            param_dict = param_dict['state_dict']

        start_with_module = False
        for k in model.state_dict().keys():
            if k.startswith('module.'):
                start_with_module = True
                break
        if start_with_module:
            param_dict = {'module.'+k : v for k, v in param_dict.items() }

        if self.rank == 0:
            print('ignore_param:')
            print([k for k, v in param_dict.items() if k not in model.state_dict() or
                    model.state_dict()[k].size() != v.size()])
            print('unload_param:')
            print([k for k, v in model.state_dict().items() if k not in param_dict.keys() or
                    param_dict[k].size() != v.size()] )

        param_dict = {k: v for k, v in param_dict.items() if k in model.state_dict() and
                        model.state_dict()[k].size() == v.size()}
        for i in param_dict:
            model.state_dict()[i].copy_(param_dict[i])

    def handle_new_batch(self):
        self.adjust_valid_alpha()
        self.scheduler_A.step()
        self.scheduler_B.step()
        self.mean_model_A.update_parameters(self.model_A)
        self.mean_model_B.update_parameters(self.model_B)

        lr = self.scheduler_A.get_lr()[0]
        alpha = self.valid_alpha

        self.batch_cnt += 1
        self.current_iteration += 1

        if (self.current_iteration > self.cfg.SOLVER.START_EVAL_STEP and self.current_iteration % self.eval_period == 0) or self.current_iteration==600:
            torch.optim.swa_utils.update_bn(self.tt_dl, self.mean_model_A)
            torch.optim.swa_utils.update_bn(self.tt_dl, self.mean_model_B)
            self.evaluate(self.mean_model_A, self.res_recoder_A)
            self.evaluate(self.mean_model_B, self.res_recoder_B)

        if self.rank==0: 
            if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
                if self.summary_writer:
                    self.summary_writer.add_scalar('Train/lr', lr, self.current_iteration)
                    self.summary_writer.add_scalar('Train/loss', self.loss_avg.avg, self.current_iteration)
            if self.current_iteration % self.cfg.SOLVER.LOG_PERIOD == 0:
                self.logger.info('Epoch[{}] Iteration[{}] Loss_A: {:.3f}, Loss_B: {:.3f}, Loss_src_A: {:.3f}, Loss_trg_A: {:.3f}, Loss_tc_A: {:.3f}, Loss_cmc_A: {:.3f}, Base Lr: {:.2e}, Alpha: {:.2f}' \
                                    .format(self.train_epoch, self.current_iteration, self.loss_A, self.loss_B, self.src_loss_A, self.trg_loss_A, \
                                        self.loss_temporal_consist_A, self.loss_cross_model_consist_A, lr, alpha))

            if self.current_iteration > self.cfg.SOLVER.START_SAVE_STEP and self.current_iteration % self.checkpoint_period == 0:
                self.save()
                self.mean_save()

    def handle_new_epoch(self):
        self.batch_cnt = 1
        if self.rank==0:
            self.logger.info('Epoch {} done'.format(self.train_epoch))
            self.logger.info('-' * 20)
        self.train_epoch += 1
    
    def adjust_valid_alpha(self, power=1.1):
        if self.current_iteration < self.cfg.SOLVER.WARMUP_STEP:
            self.valid_alpha = self.cfg.SOLVER.ALPHA_START 
        else: 
            step_region = self.cfg.SOLVER.MAX_STEPS - self.cfg.SOLVER.WARMUP_STEP
            alpha_region = self.cfg.SOLVER.ALPHA_END - self.cfg.SOLVER.ALPHA_START 
            self.valid_alpha = self.cfg.SOLVER.ALPHA_START + ((float(self.current_iteration-self.cfg.SOLVER.WARMUP_STEP)/step_region)**power)*alpha_region

    def get_threshold(self, predicted_prob, predicted_label, valid_alpha=0.2):
        thresholds = np.zeros(19)
        for c in range(19):
            x = predicted_prob[predicted_label==c]
            if len(x) == 0:
                continue        
            x = np.sort(x)
            x = x[::-1]
            thresholds[c] = max(x[np.int(np.floor(len(x)*(valid_alpha)))-1], 0)

        return thresholds

    def generate_online_psu_dis(self, trg_output_emma_A, trg_output_emma_B):
        score_A = F.softmax(trg_output_emma_A, dim=1)
        psu_value_A, psu_index_A = torch.max(score_A, dim=1)

        score_B = F.softmax(trg_output_emma_B, dim=1)
        psu_value_B, psu_index_B = torch.max(score_B, dim=1)

        if self.valid_alpha!=1.0:
            thresholds_A = self.get_threshold(psu_value_A.cpu().data[0].numpy(), psu_index_A.cpu().data[0].numpy(), valid_alpha=self.valid_alpha)
            for c in range(19):
                psu_index_A[  (psu_value_A<thresholds_A[c]) * (psu_index_A==c) ] = 255

            thresholds_B = self.get_threshold(psu_value_B.cpu().data[0].numpy(), psu_index_B.cpu().data[0].numpy(), valid_alpha=self.valid_alpha)
            for c in range(19):
                psu_index_B[  (psu_value_B<thresholds_B[c]) * (psu_index_B==c) ] = 255
        return psu_index_A, psu_index_B

    def step(self, batch):
        self.model_A.train()
        self.model_B.train()
        self.mean_model_A.train()
        self.mean_model_B.train()
        self.optim_A.zero_grad()
        self.optim_B.zero_grad()

        # source data cross entrophy
        if self.cfg.INPUT.USE_SOURCE_DATA:
            try:
                source_data, source_target, source_path = next(self.st_iter)
            except StopIteration:
                self.st_iter = iter(self.st_dl)
                source_data, source_target, source_path = next(self.st_iter)
            source_data, source_target = source_data.to(self.gpu), source_target.to(self.gpu)

            with torch.cuda.amp.autocast(enabled=self.cfg.SOLVER.MIX_PRECISION):
                src_output_A = self.model_A(source_data)
                src_output_B = self.model_B(source_data)

                self.src_loss_A = self.cross_entropy(src_output_A, source_target)
                self.src_loss_B = self.cross_entropy(src_output_B, source_target)

            self.scaler.scale(self.src_loss_A).backward()
            self.scaler.scale(self.src_loss_B).backward()


        # target data cross entrophy
        target_data, target_target, target_path = batch
        target_data, target_target = target_data.to(self.gpu), target_target.to(self.gpu)
        with torch.cuda.amp.autocast(enabled=self.cfg.SOLVER.MIX_PRECISION):
            self.loss_A = torch.tensor(0.0).to(self.gpu)
            self.loss_B = torch.tensor(0.0).to(self.gpu)

            trg_output_A = self.model_A(target_data)
            trg_output_B = self.model_B(target_data)

            self.trg_loss_A = self.cross_entropy(trg_output_A, target_target)
            self.trg_loss_B = self.cross_entropy(trg_output_B, target_target)

            self.loss_A += self.trg_loss_A
            self.loss_B += self.trg_loss_B

        # target data temporal consistency loss
            with torch.no_grad():
                trg_output_emma_A = self.mean_model_A(target_data).detach()
                trg_output_emma_B = self.mean_model_B(target_data).detach()

            self.loss_temporal_consist_A = self.consist_loss(F.softmax(trg_output_A, dim=1), F.softmax(trg_output_emma_A, dim=1))
            self.loss_temporal_consist_B = self.consist_loss(F.softmax(trg_output_B, dim=1), F.softmax(trg_output_emma_B, dim=1))

            self.loss_A += self.temporal_consist_weight * self.loss_temporal_consist_A
            self.loss_B += self.temporal_consist_weight * self.loss_temporal_consist_B

        # target data cross model consistency loss
            self.mean_model_A.eval()
            self.mean_model_B.eval()
            with torch.no_grad():
                trg_output_emma_A = self.mean_model_A(target_data).detach()
                trg_output_emma_B = self.mean_model_B(target_data).detach()
        
            online_psu_A, online_psu_B = self.generate_online_psu_dis(trg_output_emma_A, trg_output_emma_B)
            online_psu_A, online_psu_B = online_psu_A.to(self.gpu).detach(), online_psu_B.to(self.gpu).detach()

            self.loss_cross_model_consist_A = self.cross_entropy(trg_output_A, online_psu_B)
            self.loss_cross_model_consist_B = self.cross_entropy(trg_output_B, online_psu_A)

            self.loss_A += self.cross_model_consist_weight * self.loss_cross_model_consist_A
            self.loss_B += self.cross_model_consist_weight * self.loss_cross_model_consist_B

        # All loss backward
        self.scaler.scale(self.loss_A).backward()
        self.scaler.scale(self.loss_B).backward()
        
        self.scaler.step(self.optim_A)
        self.scaler.step(self.optim_B)

        self.scaler.update()

    def evaluate(self, eval_model, res_recoder):
        eval_model.eval()
        self.conf_mat = torch.zeros((self.cfg.MODEL.N_CLASS, self.cfg.MODEL.N_CLASS), dtype=torch.int64).to(self.gpu)
        with torch.no_grad():
            for batch in self.tv_dl:
                data, target, path = batch
                data = data.to(self.gpu)
                target = target.to(self.gpu)
                outputs = eval_model(data)
                loss = self.cross_entropy(outputs, target)
                target = target.data.cpu()
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    outputs = outputs[0]

                outputs = outputs.data.cpu()

                self.val_loss_avg.update(loss.cpu().item())

                _, preds = torch.max(outputs, 1)

                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                target = target.data.cpu().numpy().squeeze().astype(np.uint8)
                preds = preds[target!=255]
                target = target[target!=255]
                self.conf_mat += torch.tensor(calculate_score(self.cfg, preds, target, 'val'), dtype=torch.int64).to(self.gpu)
        dist.reduce(self.conf_mat,0, op=dist.ReduceOp.SUM)
        
        if self.rank == 0:
            val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(self.conf_mat.cpu().numpy())
            
            table = PrettyTable(["order", "name", "acc", "IoU"])
            for i in range(self.cfg.MODEL.N_CLASS):
                table.add_row([i, self.cfg.DATASETS.CLASS_NAMES[i], val_acc_per_class[i], val_IoU[i]])    
            print(table)

            valid_13_class = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
            valid_16_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
            val_13_IoU = [val_IoU[i] for i in valid_13_class]
            val_13_mean_IoU = np.mean(val_13_IoU)
            val_16_IoU = [val_IoU[i] for i in valid_16_class]
            val_16_mean_IoU = np.mean(val_16_IoU)

            table = PrettyTable(["order", "name", "acc", "IoU"])
            for i in range(self.cfg.MODEL.N_CLASS):
                table.add_row([i, self.cfg.DATASETS.CLASS_NAMES[i], val_acc_per_class[i], val_IoU[i]])

            self.logger.info('Validation Result:')
            self.logger.info(table)
            self.logger.info('Val_19_mIoU: {:.4f} | Val_16_mIoU: {:.4f} | Val_13_mIoU: {:.4f} \n'.format(
                                                                                val_mean_IoU, val_16_mean_IoU, val_13_mean_IoU))
            res_recoder.update(val_mean_IoU, self.current_iteration)
            self.logger.info(res_recoder)
            self.logger.info('-' * 20)
            if self.summary_writer:
                self.summary_writer.add_scalar('Valid/loss', self.val_loss_avg.avg, self.train_epoch)
                self.summary_writer.add_scalar('Valid/acc', np.mean(val_acc), self.train_epoch)
        dist.barrier()
        return 

    def mean_save(self):
        torch.save(self.mean_model_A.module.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_Mean_A_step' + str(self.current_iteration) + '.pth'))
        torch.save(self.mean_model_B.module.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_Mean_B_step' + str(self.current_iteration) + '.pth'))

    def save(self):
        torch.save(self.model_A.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_A_step' + str(self.current_iteration) + '.pth'))
        torch.save(self.model_B.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_B_step' + str(self.current_iteration) + '.pth'))