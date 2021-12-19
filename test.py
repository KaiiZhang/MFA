'''
Author: Kai Zhang
Date: 2021-11-29 20:52:26
LastEditors: Kai Zhang
LastEditTime: 2021-11-30 16:22:00
Description: test
'''

from torch.nn.functional import softmax
from tqdm import tqdm
import argparse
from dataset import make_dataloader
import os

def evaluate(model, dataloader, args, cfg, gpu, flip=True, save_res=False):
    import numpy as np
    import torch
    from utils import calculate_score, m_evaluate
    from prettytable import PrettyTable
    from PIL import Image

    model.eval()
    conf_mat = np.zeros((cfg.MODEL.N_CLASS, cfg.MODEL.N_CLASS)).astype(np.int64)
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data, target, path = batch
            data = data.to(gpu)
            # data = torch.flip(data, (1,))  # in old version input channel is BGR, but now it is RGB
            outputs = softmax(model(data, upsample=True))
            if flip:
                outputs += softmax(torch.flip(model(torch.flip(data, (3,))), (3,)))
            outputs = outputs.data.cpu()
            _, preds = torch.max(outputs, (1))
            # save result
            if save_res:
                for i, pred in enumerate(preds):
                    output_nomask = np.asarray( pred.cpu().numpy(), dtype=np.uint8 )
                    output_nomask = Image.fromarray(output_nomask)  
                    name = path[i].split('/')[-1]
                    output_nomask.save( '%s/%s' % (args.output, name))
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            target = target.data.cpu().numpy().squeeze().astype(np.uint8)
            
            preds = preds[target!=255]
            target = target[target!=255]
            conf_mat = conf_mat + calculate_score(cfg, preds, target, 'val')
            
    val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(conf_mat)
    if cfg.TEST.ONLY13_CLASSES:
        valid_class = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        val_IoU = [val_IoU[i] for i in valid_class]
        val_mean_IoU = np.mean(val_IoU)
        table = PrettyTable(["order", "name", "acc", "IoU"])
        for i in range(13):
            ind = valid_class[i]
            table.add_row([i, cfg.DATASETS.CLASS_NAMES[ind], val_acc_per_class[ind], val_IoU[i]])
    else:
        table = PrettyTable(["order", "name", "acc", "IoU"])
        for i in range(cfg.MODEL.N_CLASS):
            table.add_row([i, cfg.DATASETS.CLASS_NAMES[i], val_acc_per_class[i], val_IoU[i]])    
    print(table)
    print('VAL_ACC: %s VAL_MEAN_IOU: %s VAL_KAPPA: %s \n' %
                         (val_acc, val_mean_IoU, val_kappa))

def test_model(target_val_loader, args, cfg, gpu = 0):
    from model import build_model
    from torchsummary import summary
    import torch

    print('==========> start test model')
    model = build_model(cfg)
    # summary(model.cuda(), (3, 1024, 2048))
    print('load from: ', cfg.TEST.WEIGHT)
    param_dict = torch.load(cfg.TEST.WEIGHT, map_location=lambda storage, loc: storage)
    if 'state_dict' in param_dict.keys():
        param_dict = param_dict['state_dict']

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
    model = model.to(gpu)
    evaluate(model, target_val_loader, args, cfg, gpu)

if __name__ == '__main__':
    from config import cfg

    parser = argparse.ArgumentParser(description="Baseline Testing")
    parser.add_argument("--config_file", default="./configs/mfa.yml", help="path to config file", type=str)
    parser.add_argument("-g", "--gpu", default=0, help="which gpu to use", type=int)
    parser.add_argument("--output", default="./outputs",help="dir to save inference results", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    _, _, target_val_loader = make_dataloader(cfg, 0)
    test_model(target_val_loader, args, cfg, gpu=args.gpu)
