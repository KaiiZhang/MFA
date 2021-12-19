'''
Author: Kai Zhang
Date: 2021-11-29 13:27:56
LastEditors: Kai Zhang
LastEditTime: 2021-11-29 16:03:58
Description: Unit Testing
'''
from config import cfg
from tqdm import tqdm
config_file = './configs/mfa_syn.yml'
# config_file = './configs/mfa.yml'
cfg.merge_from_file(config_file)


def test_dataset():
    '''
    dataset 的单元测试
    '''
    from dataset import make_dataloader
    print('==========> start test dataset')
    source_train_loader, target_train_loader, target_val_loader = make_dataloader(cfg, 0)
    source_train_loader_iter = iter(source_train_loader)
    img, gt, path = next(source_train_loader_iter)
    print('Source train:')
    print(img.shape)
    print(gt.shape)
    print(path)

    target_train_loader_iter = iter(target_train_loader)
    img_, gt_, path_ = next(target_train_loader_iter)
    print('Target train:')
    print(img_.shape)
    print(gt_.shape)
    print(path_)

    target_val_loader_iter = iter(target_val_loader)
    img, gt, path = next(target_val_loader_iter)
    print('Target val:')
    print(img.shape)
    print(gt.shape)
    print(path)
    print('==========> test dataset sucess!')

    return target_val_loader

def evaluate(model, dataloader, gpu):
    from tqdm import tqdm 
    import numpy as np
    import torch
    from utils import calculate_score, m_evaluate
    from prettytable import PrettyTable
    from torch import nn
    from PIL import Image

    model.eval()
    conf_mat = np.zeros((cfg.MODEL.N_CLASS, cfg.MODEL.N_CLASS)).astype(np.int64)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            data, target, path = batch
            data = data.to(gpu)
            outputs = model(data, upsample=True)
            outputs = outputs.data.cpu()
            _, preds = torch.max(outputs, 1)
            # save result
            # for i, pred in enumerate(preds):
            #     output_nomask = np.asarray( pred.cpu().numpy(), dtype=np.uint8 )
            #     # print(output_nomask.shape)
            #     # print(type(output_nomask))
            #     # print(np.unique(output_nomask))
            #     output_nomask = Image.fromarray(output_nomask)  
            #     name = path[i].split('/')[-1]
            #     output_nomask.save( '%s/%s' % ('./result', name))
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            target = target.data.cpu().numpy().squeeze().astype(np.uint8)
            
            preds = preds[target!=255]
            target = target[target!=255]
            conf_mat = conf_mat + calculate_score(cfg, preds, target, 'val')

    if cfg.TEST.ONLY13_CLASSES:
        conf_mat = conf_mat[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]
        conf_mat = conf_mat[:, [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(conf_mat)
        table = PrettyTable(["order", "name", "acc", "IoU"])
        for i in range(13):
            ind = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18][i]
            table.add_row([i, cfg.DATASETS.CLASS_NAMES[ind], val_acc_per_class[i], val_IoU[i]])
    else:
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(conf_mat)
        table = PrettyTable(["order", "name", "acc", "IoU"])
        for i in range(cfg.MODEL.N_CLASS):
            table.add_row([i, cfg.DATASETS.CLASS_NAMES[i], val_acc_per_class[i], val_IoU[i]])    
    print(table)
    print('VAL_ACC: %s VAL_MEAN_IOU: %s VAL_KAPPA: %s \n' %
                         (val_acc, val_mean_IoU, val_kappa))

def test_model(target_val_loader, gpu = 0):
    from model import build_model
    from torchsummary import summary
    import torch
    '''
    model 的单元测试
    '''
    print('==========> start test model')
    model = build_model(cfg)
    summary(model.cuda(), (3, 512, 512))
    if cfg.SOLVER.RESUME:
        param_dict = torch.load(cfg.SOLVER.RESUME_CHECKPOINT_B, map_location=lambda storage, loc: storage)
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
    evaluate(model, target_val_loader, gpu)

if __name__ == '__main__':
    target_val_loader = test_dataset()
    test_model(target_val_loader, gpu=0)
