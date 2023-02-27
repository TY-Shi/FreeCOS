from __future__ import division
import os.path as osp
import os
import sys
import time
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader,get_val_loader
from network import Network, Network_UNet,SingleUNet_featureDA, Single_contrast_UNet
from dataloader import XCAD
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from utils.evaluation_metric import computeF1, compute_allXCAD, compute_allRetinal
from Datasetloader.dataset import CSDataset
from common.logger import Logger
import csv
import PIL.Image as Image
from network import SingleUNet

def get_parser():
    parser = argparse.ArgumentParser(description='JTFN for Curvilinear Structure Segmentation')
    parser.add_argument('--config', type=str, default='config/UNet_DRIVE.yaml', help='Model config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def create_csv(path, csv_head):
    # path = "aa.csv"
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)


def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)


def main():
    if os.getenv('debug') is not None:
        is_debug = os.environ['debug']
    else:
        is_debug = False
    parser = argparse.ArgumentParser()
    os.environ['MASTER_PORT'] = '169711'


    args = parser.parse_args()
    cudnn.benchmark = True
    #set seed
    seed = config.seed #12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("Begin Dataloader.....")

    CSDataset.initialize(datapath=config.datapath)
    dataloader_val = CSDataset.build_dataloader(config.benchmark,
                                                config.batch_size_val,
                                                config.nworker,
                                                'val',
                                                'same',
                                                None,
                                                'supervised')
    print("Dataloader.....")
    #model = SingleUNet(1,config.num_classes)
    #model = SingleUNet(4, config.num_classes)
    model = Single_contrast_UNet(4, config.num_classes)
    #model = SingleUNet_featureDA(4, config.num_classes)
    #model = Network_UNet(1,config.num_classes)
    print('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = model.cuda()
        model = nn.DataParallel(model)
        print('Use GPU Parallel.')
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model

    mode_dict = {}
    if config.model_weight:
        if os.path.isfile(config.model_weight):
            print("=> loading weight '{}'".format(config.model_weight))
            checkpoint = torch.load(config.model_weight)
            for k,v in checkpoint['state_dict'].items():
                print("k",k)
                k_replace_list = k.split(".")
                new_list = k_replace_list[1:]
                #new_list = 'module.' + k
                new_list = k
                print("new_list",new_list)
                #new_key = '.'.join(str(i) for i in new_list)
                new_key = new_list
                print("new_key",new_key)
                mode_dict[new_key] = checkpoint['state_dict'][k]

            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(mode_dict)
            print("=> loaded weight '{}'".format(config.model_weight))
        else:
            print("=> no weight found at '{}'".format(config.model_weight))
    else:
        raise RuntimeError("Please support weight.")
    best_val_f1 = 0
    best_thresh = 0
    best_val_AUC = 0
    different_thresh_score = os.path.join('logs', config.logname + '.log') + '/' + 'different_thresh_{}.csv'.format(config.benchmark)
    csv_head = ["Threshold", "f1", "pr", "recall","Sp","Acc","JC","AUC"]
    create_csv(different_thresh_score, csv_head)
    with torch.no_grad():
        #val_mean_f1, val_mean_pr, val_mean_re, val_mean_sp, val_mean_acc, val_mean_jc, val_mean_AUC = evaluate(model, dataloader_val)
        for threshold in range(1, 100):
            thresh = threshold / 100
            val_mean_f1, val_mean_pr, val_mean_re, val_mean_sp, val_mean_acc, val_mean_jc, val_mean_AUC = evaluate_thresh(model, dataloader_val,thresh)
            # print("val_mean_f1",val_mean_f1)
            # print("val_mean_re",val_mean_re)
            if val_mean_f1>best_val_f1:
                best_val_f1 = val_mean_f1
                best_thresh = thresh
                print("best_val_f1",best_val_f1)
            data_row_f1score = [str(thresh), str(val_mean_f1.item()), str(val_mean_pr.item()), str(val_mean_re.item()), str(val_mean_sp.item()), str(val_mean_acc.item()), str(val_mean_jc),str(val_mean_AUC)]
            write_csv(different_thresh_score, data_row_f1score)
        val_mean_f1, val_mean_pr, val_mean_re, val_mean_sp, val_mean_acc, val_mean_jc, val_mean_AUC = evaluate_thresh(model,dataloader_val,best_thresh)
    print("best_thresh:{}".format(best_thresh))
    print('F1: {:.2f} Precision: {:.2f} Recall: {:.2f}'.format(val_mean_f1.item(), val_mean_pr.item(), val_mean_re.item()))
    print('Sp: {:.2f} Acc: {:.2f} JC: {:.2f}'.format(val_mean_sp.item(), val_mean_acc.item(), val_mean_jc))
    print('AUC: {}'.format(val_mean_AUC))
    print('==================== Finished Testing ====================')


def save_image(prob, minibatch, save_path, idx):
    #prob NCHW
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    out = prob[0, 0, :, :].cpu().numpy()
    name = minibatch['img_name'][0]
    img = minibatch['img'][0, 0, :, :].cpu().numpy()
    anno_mask = minibatch['anno_mask'][0, 0, :, :].cpu().numpy()
    #print(name, out.shape, img.shape, anno_mask.shape)
    im = Image.fromarray(img * 255.).convert("L")
    im_name = save_path + '{}_{}_image.png'.format(str(idx), name)
    im.save(im_name)
    im = Image.fromarray(anno_mask * 255.).convert("L")
    im_name = save_path + '{}_{}_mask.png'.format(str(idx), name)
    im.save(im_name)
    im = Image.fromarray(out * 255.).convert("L")
    im_name = save_path + '{}_{}_pred.png'.format(str(idx), name)
    im.save(im_name)

def evaluate(model, dataloader):
    # Force randomness during training / freeze randomness during testing
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    val_sum_f1 = 0
    val_sum_pr = 0
    val_sum_re = 0
    save_path = os.path.join('logs', config.logname + '.log') + '/save_image_{}/'.format(config.benchmark)
    val_score_path = os.path.join('logs', config.logname + '.log') + '/' + 'val_train_f1_singleval_{}.csv'.format(config.benchmark)
    csv_head = ["image_name", "f1", "pr", "recall","Sp","Acc","JC","AUC"]
    create_csv(val_score_path, csv_head)
    val_sum_sp = 0
    val_sum_acc = 0
    val_sum_jc = 0
    val_sum_AUC = 0
    for idx, minibatch in enumerate(dataloader):
        # 1. Forward pass
        val_imgs = minibatch['img']
        val_gts = minibatch['anno_mask']
        val_mask = minibatch['ignore_mask']
        val_imgs = val_imgs.cuda(non_blocking=True)
        val_gts = val_gts.cuda(non_blocking=True)
        val_mask = val_mask.cuda(non_blocking=True)

        val_pred_sup_l = model(val_imgs)  # N1HW
        #val_pred_sup_l = model(val_imgs, step=1)  # N1HW
        pred_mask = torch.where(val_pred_sup_l >= 0.5, 1, 0)
        #val_f1, val_precision, val_recall = computeF1(pred_mask, val_gts)
        #val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc = compute_allXCAD(pred_mask, val_gts)
        #print("val_mask",torch.unique(val_mask))
        eva_binary = pred_mask*val_mask
        eva_prob = val_pred_sup_l*val_mask
        val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc, val_AUC = compute_allRetinal(eva_binary, eva_prob, val_gts)
        # f1, precision, recall, f1_thinlist,precision_thinlist,recall_thinlist,f1_thicklist,precision_thick,recall_thick, quality, cor, com
        img_name = minibatch['img_name']
        save_image(val_pred_sup_l, minibatch, save_path, idx)
        data_row_f1score = [str(img_name), str(val_f1), str(val_precision), str(val_recall), str(val_Sp), str(val_Acc), str(val_jc),str(val_AUC)]
        write_csv(val_score_path, data_row_f1score)
        val_sum_f1 += val_f1
        val_sum_pr += val_precision
        val_sum_re += val_recall

        val_sum_sp += val_Sp
        val_sum_acc += val_Acc
        val_sum_jc += val_jc
        val_sum_AUC += val_AUC
    val_mean_f1 = val_sum_f1 / len(dataloader)
    val_mean_pr = val_sum_pr / len(dataloader)
    val_mean_re = val_sum_re / len(dataloader)
    val_mean_sp = val_sum_sp / len(dataloader)
    val_mean_acc = val_sum_acc / len(dataloader)
    val_mean_jc = val_sum_jc / len(dataloader)
    val_mean_AUC = val_sum_AUC / len(dataloader)

    return val_mean_f1, val_mean_pr, val_mean_re, val_mean_sp, val_mean_acc, val_mean_jc, val_mean_AUC


def evaluate_thresh(model, dataloader,thresh):
    # Force randomness during training / freeze randomness during testing
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    val_sum_f1 = 0
    val_sum_pr = 0
    val_sum_re = 0
    save_path = os.path.join('logs', config.logname + '.log') + '/save_image__thresh{}/'.format(config.benchmark)
    val_score_path = os.path.join('logs', config.logname + '.log') + '/' + 'val_train_f1_singleval_thresh_{}.csv'.format(config.benchmark)
    csv_head = ["image_name", "f1", "pr", "recall","Sp","Acc","JC","AUC"]
    create_csv(val_score_path, csv_head)
    val_sum_sp = 0
    val_sum_acc = 0
    val_sum_jc = 0
    val_sum_AUC = 0
    for idx, minibatch in enumerate(dataloader):
        # 1. Forward pass
        val_imgs = minibatch['img']
        val_gts = minibatch['anno_mask']
        val_imgs = val_imgs.cuda(non_blocking=True)
        val_gts = val_gts.cuda(non_blocking=True)

        print(torch.max(val_imgs), torch.min(val_imgs))

        #val_pred_sup_l = model(val_imgs)  # N1HW
        val_pred_sup_l, sample_set_unsup, _ = model(val_imgs, mask=None, trained=False, fake=False)
        pred_mask = torch.where(val_pred_sup_l >= thresh, 1, 0)

        #val_f1, val_precision, val_recall = computeF1(pred_mask, val_gts)
        #val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc = compute_allXCAD(pred_mask, val_gts)
        #print("val_mask",torch.unique(val_mask))
        eva_binary = pred_mask
        eva_prob = val_pred_sup_l
        val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc, val_AUC = compute_allRetinal(eva_binary, eva_prob, val_gts)
        # f1, precision, recall, f1_thinlist,precision_thinlist,recall_thinlist,f1_thicklist,precision_thick,recall_thick, quality, cor, com
        img_name = minibatch['img_name']
        save_image(val_pred_sup_l, minibatch, save_path, idx)
        data_row_f1score = [str(img_name), str(val_f1), str(val_precision), str(val_recall), str(val_Sp), str(val_Acc), str(val_jc),str(val_AUC)]
        write_csv(val_score_path, data_row_f1score)
        val_sum_f1 += val_f1
        val_sum_pr += val_precision
        val_sum_re += val_recall

        val_sum_sp += val_Sp
        val_sum_acc += val_Acc
        val_sum_jc += val_jc
        val_sum_AUC += val_AUC
    val_mean_f1 = val_sum_f1 / len(dataloader)
    val_mean_pr = val_sum_pr / len(dataloader)
    val_mean_re = val_sum_re / len(dataloader)
    val_mean_sp = val_sum_sp / len(dataloader)
    val_mean_acc = val_sum_acc / len(dataloader)
    val_mean_jc = val_sum_jc / len(dataloader)
    val_mean_AUC = val_sum_AUC / len(dataloader)

    return val_mean_f1, val_mean_pr, val_mean_re, val_mean_sp, val_mean_acc, val_mean_jc, val_mean_AUC

if __name__ == '__main__':
    main()
