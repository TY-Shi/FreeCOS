# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345  #3407

remoteip = os.popen('pwd').read()
if os.getenv('volna') is not None:
    C.volna = os.environ['volna']
else:
    C.volna = '/data/sty/Unsupervised/' # the path to the data dir.

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'TorchSemiSeg'
C.datapath = "/data/sty/Unsupervised_dxh/Data/XCAD"
C.benchmark = 'XCAD_LIOT'#XCAD_LIOT
C.logname = 'test_XCADmodel'
#next experiment memory
C.log_dir = "/data/sty/Unsupervised_dxh"
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.dataset_path = C.volna + "Data/CrackTree"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_img_root_folder = C.dataset_path + "/train/img"
C.train_fakegt_root_folder = C.dataset_path + "/train/fake_vessel"
C.val_img_root_folder = C.dataset_path + "/test/img"
C.val_gt_root_folder = C.dataset_path + "/test/gt"
C.pretrained_model = C.volna +  'DATA/pytorch-weight/resnet50_v1c.pth'

# model path
C.model_weight = '/data/sty/Unsupervised_dxh/logs/contrast_repeatnce4_negall_Nce_allpixel_noise.log/best_Segment.pt'

"""Path Config"""


''' Experiments Setting '''
C.labeled_ratio = 1
C.train_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))
C.eval_source = osp.join(C.dataset_path, "config_new/val.txt")
C.test_source = osp.join(C.dataset_path, "config_new/test.txt")
C.demo_source = osp.join(C.dataset_path, "config_new/demo.txt")

C.is_test = False
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

C.cps_weight = 6


"""Image Config"""
C.num_classes = 1
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.image_height = 256
C.image_width = 256
C.num_train_imgs = 150 // C.labeled_ratio #XCAD
C.num_eval_imgs = 126#XCAD
C.num_unsup_imgs = 1621 #XCAD

"""Train Config"""
C.lr = 0.01
C.lr_D = 0.001
C.batch_size = 8
C.batch_size_val = 1
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.state_epoch = 0
C.nepochs = 1000
C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)
C.cold_start = 0
C.niters_per_epoch = C.max_samples // C.batch_size

C.nworker = 8
C.img_mode = 'crop'
C.img_size = 256
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_base_size = 800
C.eval_crop_size = 800

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50
C.warm_up_epoch = 0

