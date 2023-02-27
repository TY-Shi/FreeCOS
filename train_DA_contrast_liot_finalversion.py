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
from dataloader import get_train_loader, get_val_loader
from network import Network, Network_UNet, SingleUNet, Single_IBNUNet, Single_contrast_UNet
from dataloader import XCAD
from utils.init_func import init_weight, group_weight
from utils.data_disturb import noise_input
from engine.lr_policy import WarmUpPolyLR, CosinLR
from utils.evaluation_metric import computeF1, compute_allRetinal
from Datasetloader.dataset import CSDataset
from common.logger import Logger
import csv
from utils.loss_function import DiceLoss, Contrastloss, ContrastRegionloss, ContrastRegionloss_noedge, \
    ContrastRegionloss_supunsup, ContrastRegionloss_NCE, ContrastRegionloss_AllNCE, ContrastRegionloss_quaryrepeatNCE, Triplet
import copy
from base_model.discriminator import PredictDiscriminator, PredictDiscriminator_affinity
from base_model.feature_memory import FeatureMemory_TWODomain_NC
import numpy as np

def create_csv(path, csv_head):
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

def check_feature(sample_set_sup, sample_set_unsup):
    """
    feature N,dims，Has bug or debuff because of zeros
    """
    flag = True
    Single = False
    queue_len = 500
    # sample_set_sup['sample_easy_pos'], sample_set_sup['sample_easy_neg'], sample_set_unsup['sample_easy_pos'], sample_set_unsup['sample_easy_neg']
    with torch.no_grad():
        if 'sample_easy_pos' not in sample_set_sup.keys() or 'sample_easy_neg' not in sample_set_unsup.keys() or 'sample_easy_pos' not in sample_set_unsup.keys():
            flag = False
            quary_feature = None
            pos_feature = None
            neg_feature = None
        else:
            quary_feature = sample_set_sup['sample_easy_pos']
            pos_feature = sample_set_unsup['sample_easy_pos']
            neg_feature = sample_set_unsup['sample_easy_neg']
            flag = True

        if 'sample_easy_neg' in sample_set_sup.keys() and 'sample_easy_neg' in sample_set_unsup.keys():
            neg_unlabel = sample_set_unsup['sample_easy_neg']
            neg_label = sample_set_sup['sample_easy_neg']
            neg_feature = torch.cat((neg_unlabel[:min(queue_len // 2, neg_unlabel.shape[0]), :],
                                     neg_label[:min(queue_len // 2, neg_label.shape[0]), :]), dim=0)
    return quary_feature, pos_feature, neg_feature, flag

def train(epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
          optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
          average_negregion):
    if torch.cuda.device_count() > 1:
        Segment_model.module.train()
        predict_Discriminator_model.module.train()
    else:
        print("start_model_train")
        Segment_model.train()
        predict_Discriminator_model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
    dataloader = iter(dataloader_supervised)
    unsupervised_dataloader = iter(dataloader_unsupervised)
    bce_loss = nn.BCELoss()
    sum_loss_seg = 0
    sum_adv_loss = 0
    sum_Dsrc_loss = 0
    sum_Dtar_loss = 0
    sum_totalloss = 0
    sum_contrastloss = 0
    sum_celoss = 0
    sum_diceloss = 0
    source_label = 0
    target_label = 1
    criterion_contrast = ContrastRegionloss_quaryrepeatNCE()
    print('begin train')
    ''' supervised part '''
    for idx in pbar:
        current_idx = epoch * config.niters_per_epoch + idx
        damping = (1 - current_idx / total_iteration)
        start_time = time.time()
        optimizer_l.zero_grad()
        optimizer_D.zero_grad()
        try:
            minibatch = dataloader.next()
        except StopIteration:
            dataloader = iter(dataloader_supervised)
            minibatch = dataloader.next()

        imgs = minibatch['img']
        gts = minibatch['anno_mask']
        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        with torch.no_grad():
            weight_mask = gts.clone().detach()
            weight_mask[weight_mask == 0] = 0.1
            weight_mask[weight_mask == 1] = 1
            criterion_bce = nn.BCELoss(weight=weight_mask)
        try:
            unsup_minibatch = unsupervised_dataloader.next()
        except StopIteration:
            unsupervised_dataloader = iter(dataloader_unsupervised)
            unsup_minibatch = unsupervised_dataloader.next()

        unsup_imgs = unsup_minibatch['img']
        unsup_imgs = unsup_imgs.cuda(non_blocking=True)

        # Start train fake vessel
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = False
        pred_sup_l, sample_set_sup, flag_sup = Segment_model(imgs, mask=gts, trained=True, fake=True)
        loss_ce = 0.1 * criterion_bce(pred_sup_l, gts)  # For retinal :5 For XCAD:0.1 5 for crack
        loss_dice = criterion(pred_sup_l, gts)
        pred_target, sample_set_unsup, flag_un = Segment_model(unsup_imgs, mask=None, trained=True, fake=False)
        D_seg_target_out = predict_Discriminator_model(pred_target)
        loss_adv_target = bce_loss(F.sigmoid(D_seg_target_out),
                                   torch.FloatTensor(D_seg_target_out.data.size()).fill_(source_label).cuda())
        quary_feature, pos_feature, neg_feature, flag = check_feature(sample_set_sup, sample_set_unsup)

        if flag:
            loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature)
        else:
            loss_contrast = 0
        weight_contrast = 0.04  # 0.04 for NCE allpixel/0.01maybe same as dice
        loss_seg = loss_dice + loss_ce
        sum_loss_seg += loss_seg.item()
        loss_contrast_sum = weight_contrast * (loss_contrast)
        sum_contrastloss += loss_contrast_sum

        loss_adv = (loss_adv_target * damping) / 4 + loss_dice + loss_ce + weight_contrast * (loss_contrast)
        loss_adv.backward(retain_graph=False)
        loss_adv_sum = (loss_adv_target * damping) / 4
        sum_adv_loss += loss_adv_sum.item()

        sum_celoss += loss_ce
        sum_diceloss += loss_dice.item()
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = True
        pred_sup_l = pred_sup_l.detach()
        D_out_src = predict_Discriminator_model(pred_sup_l)

        loss_D_src = bce_loss(F.sigmoid(D_out_src), torch.FloatTensor(
            D_out_src.data.size()).fill_(source_label).cuda())
        loss_D_src = loss_D_src / 8
        loss_D_src.backward(retain_graph=False)
        sum_Dsrc_loss += loss_D_src.item()

        pred_target = pred_target.detach()
        D_out_tar = predict_Discriminator_model(pred_target)

        loss_D_tar = bce_loss(F.sigmoid(D_out_tar), torch.FloatTensor(
            D_out_tar.data.size()).fill_(target_label).cuda())
        loss_D_tar = loss_D_tar / 8  # bias
        loss_D_tar.backward(retain_graph=False)
        sum_Dtar_loss += loss_D_tar.item()
        optimizer_l.step()
        optimizer_D.step()

        lr = lr_policy.get_lr(current_idx)  # lr change
        optimizer_l.param_groups[0]['lr'] = lr
        optimizer_l.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer_l.param_groups)):
            optimizer_l.param_groups[i]['lr'] = lr

        Lr_D = lrD_policy.get_lr(current_idx)
        optimizer_D.param_groups[0]['lr'] = Lr_D
        for i in range(2, len(optimizer_D.param_groups)):
            optimizer_D.param_groups[i]['lr'] = Lr_D

        sum_contrastloss += loss_contrast_sum
        print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                    + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.2e' % lr \
                    + ' loss_seg=%.4f' % loss_seg.item() \
                    + ' loss_D_tar=%.4f' % loss_D_tar.item() \
                    + ' loss_D_src=%.4f' % loss_D_src.item() \
                    + ' loss_adv=%.4f' % loss_adv.item() \
                    + 'loss_ce=%.4f' % loss_ce \
                    + 'loss_dice=%.4f' % loss_dice.item() \
                    + 'loss_contrast=%.4f' % loss_contrast_sum

        sum_totalloss = sum_totalloss + sum_Dtar_loss + sum_Dsrc_loss + sum_adv_loss + sum_loss_seg + sum_contrastloss
        pbar.set_description(print_str, refresh=False)

        end_time = time.time()

    train_loss_seg = sum_loss_seg / len(pbar)
    train_loss_Dtar = sum_Dtar_loss / len(pbar)
    train_loss_Dsrc = sum_Dsrc_loss / len(pbar)
    train_loss_adv = sum_adv_loss / len(pbar)
    train_loss_ce = sum_celoss / len(pbar)
    train_loss_dice = sum_diceloss / len(pbar)
    train_loss_contrast = sum_contrastloss / len(pbar)
    train_total_loss = train_loss_seg + train_loss_Dtar + train_loss_Dsrc + train_loss_adv + train_loss_contrast
    return train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion


# evaluate(epoch, model, dataloader_val,criterion,criterion_consist)
def evaluate(epoch, Segment_model, predict_Discriminator_model, val_target_loader, criterion):
    if torch.cuda.device_count() > 1:
        Segment_model.module.eval()
        predict_Discriminator_model.module.eval()
    else:
        Segment_model.eval()
        predict_Discriminator_model.eval()
    with torch.no_grad():
        val_sum_loss_sup = 0
        val_sum_f1 = 0
        val_sum_pr = 0
        val_sum_re = 0
        val_sum_sp = 0
        val_sum_acc = 0
        val_sum_jc = 0
        val_sum_AUC = 0
        F1_best = 0
        print('begin eval')
        ''' supervised part '''
        for val_idx, minibatch in enumerate(val_target_loader):
            start_time = time.time()
            val_imgs = minibatch['img']
            val_gts = minibatch['anno_mask']
            val_imgs = val_imgs.cuda(non_blocking=True)
            val_gts = val_gts.cuda(non_blocking=True)
            # NCHW
            val_pred_sup_l, sample_set_unsup, _ = Segment_model(val_imgs, mask=None, trained=False, fake=False)

            max_l = torch.where(val_pred_sup_l >= 0.5, 1, 0)
            val_max_l = max_l.float()
            val_loss_sup = criterion(val_pred_sup_l, val_gts)

            current_validx = epoch * config.niters_per_epoch + val_idx
            val_loss = val_loss_sup
            val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc, val_AUC = compute_allRetinal(val_max_l,
                                                                                                     val_pred_sup_l,
                                                                                                     val_gts)
            val_sum_loss_sup += val_loss_sup.item()
            val_sum_f1 += val_f1
            val_sum_pr += val_precision
            val_sum_re += val_recall
            val_sum_AUC += val_AUC
            val_sum_sp += val_Sp
            val_sum_acc += val_Acc
            val_sum_jc += val_jc
            end_time = time.time()
        val_mean_f1 = val_sum_f1 / len(val_target_loader)
        val_mean_pr = val_sum_pr / len(val_target_loader)
        val_mean_re = val_sum_re / len(val_target_loader)
        val_mean_AUC = val_sum_AUC / len(val_target_loader)
        val_mean_acc = val_sum_acc / len(val_target_loader)
        val_mean_sp = val_sum_sp / len(val_target_loader)
        val_mean_jc = val_sum_jc / len(val_target_loader)
        val_loss_sup = val_sum_loss_sup / len(val_target_loader)
        return val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup


def main():
    if os.getenv('debug') is not None:
        is_debug = os.environ['debug']
    else:
        is_debug = False
    parser = argparse.ArgumentParser()
    os.environ['MASTER_PORT'] = '169711'

    args = parser.parse_args()
    cudnn.benchmark = True
    # set seed
    seed = config.seed  # 12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("Begin Dataloader.....")

    CSDataset.initialize(datapath=config.datapath)
    dataloader_supervised = CSDataset.build_dataloader(config.benchmark,  # XCAD
                                                       config.batch_size,
                                                       config.nworker,
                                                       'train',
                                                       config.img_mode,
                                                       config.img_size,
                                                       'supervised')  # FDA sys
    dataloader_unsupervised = CSDataset.build_dataloader(config.benchmark,  # XCAD
                                                         config.batch_size,
                                                         config.nworker,
                                                         'train',
                                                         config.img_mode,
                                                         config.img_size,
                                                         'unsupervised')

    dataloader_val = CSDataset.build_dataloader(config.benchmark,
                                                config.batch_size_val,
                                                config.nworker,
                                                'val',
                                                'same',
                                                None,
                                                'supervised')
    print("Dataloader.....")
    criterion = DiceLoss()  # try both loss BCE and DICE

    # define and init the model
    # Single or not single
    BatchNorm2d = nn.BatchNorm2d
    Segment_model = Single_contrast_UNet(4, config.num_classes)

    init_weight(Segment_model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    # define the learning rate
    base_lr = config.lr  # 0.04
    base_lr_D = config.lr_D  # 0.04

    params_list_l = []
    params_list_l = group_weight(params_list_l, Segment_model.backbone,
                                 BatchNorm2d, base_lr)
    # optimizer for segmentation_L
    print("config.weight_decay", config.weight_decay)
    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    predict_Discriminator_model = PredictDiscriminator(num_classes=1)
    init_weight(predict_Discriminator_model, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    optimizer_D = torch.optim.Adam(predict_Discriminator_model.parameters(),
                                   lr=base_lr_D, betas=(0.9, 0.99))

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch  # nepochs=137  niters=C.max_samples // C.batch_size
    print("total_iteration", total_iteration)
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    lrD_policy = WarmUpPolyLR(base_lr_D, config.lr_power, total_iteration,
                              config.niters_per_epoch * config.warm_up_epoch)

    average_posregion = torch.zeros((1, 128))
    average_negregion = torch.zeros((1, 128))
    if torch.cuda.device_count() > 1:
        Segment_model = Segment_model.cuda()
        Segment_model = nn.DataParallel(Segment_model)
        average_posregion.cuda()
        average_negregion.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda()
        predict_Discriminator_model = nn.DataParallel(predict_Discriminator_model)
        # Logger.info('Use GPU Parallel.')
    elif torch.cuda.is_available():
        print("cuda_is available")
        Segment_model = Segment_model.cuda()
        average_posregion.cuda()
        average_negregion.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda()
    else:
        Segment_model = Segment_model
        predict_Discriminator_model = predict_Discriminator_model

    best_val_f1 = 0
    best_val_AUC = 0
    Logger.initialize(config, training=True)
    val_score_path = os.path.join('logs', config.logname + '.log') + '/' + 'val_train_f1.csv'
    csv_head = ["epoch", "total_loss", "f1", "AUC", "pr", "recall", "Acc", "Sp", "JC"]
    create_csv(val_score_path, csv_head)
    for epoch in range(config.state_epoch, config.nepochs):
        # train_loss_sup, train_loss_consis, train_total_loss
        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion = train(
            epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
            optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
            average_negregion)
        print(
            "train_seg_loss:{},train_loss_Dtar:{},train_loss_Dsrc:{},train_loss_adv:{},train_total_loss:{},train_loss_contrast:{}".format(
                train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss,
                train_loss_contrast))
        print("train_loss_dice:{},train_loss_ce:{}".format(train_loss_dice, train_loss_ce))
        # val_mean_f1, val_mean_pr, val_mean_re, val_mean_f1, val_mean_pr, val_mean_re,val_loss_sup
        val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup = evaluate(
            epoch, Segment_model, predict_Discriminator_model, dataloader_val,
            criterion)  # evaluate(epoch, model, val_target_loader,criterion, criterion_cps)
        # val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re,val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup
        data_row_f1score = [str(epoch), str(train_total_loss), str(val_mean_f1.item()), str(val_mean_AUC),
                            str(val_mean_pr.item()), str(val_mean_re.item()), str(val_mean_acc), str(val_mean_sp),
                            str(val_mean_jc)]
        print("val_mean_f1", val_mean_f1.item())
        print("val_mean_AUC", val_mean_AUC)
        print("val_mean_pr", val_mean_pr.item())
        print("val_mean_re", val_mean_re.item())
        print("val_mean_acc", val_mean_acc.item())
        write_csv(val_score_path, data_row_f1score)
        if val_mean_f1 > best_val_f1:
            best_val_f1 = val_mean_f1
            Logger.save_model_f1_S(Segment_model, epoch, val_mean_f1, optimizer_l)
            Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_f1, optimizer_D)
        # if val_mean_AUC > best_val_AUC:
        #     best_val_AUC = val_mean_AUC
        #     Logger.save_model_f1_S(Segment_model, epoch, val_mean_AUC, optimizer_l)
        #     Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_AUC, optimizer_D)


if __name__ == '__main__':
    main()
