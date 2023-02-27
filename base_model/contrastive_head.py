import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
import numpy as np
import PIL.Image as Image
#from base_model.contrastive_utils import get_query_keys, get_query_keys_eval, enhance_op
from utils.contrastive_utils import get_query_keys, get_query_keys_eval, enhance_op, get_query_keys_sty, get_query_keys_myself
#from contrastive_utils import get_query_keys, get_query_keys_eval, enhance_op, get_query_keys_myself
from vlkit.dense import (
    seg2edge as vlseg2edge,
    sobel, flux2angle, dense2flux, quantize_angle
    )


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

@HEADS.register_module()
class ContrastiveHead(BaseModule):

    def __init__(self,
                 num_convs=4,
                 num_projectfc=2,
                 roi_feat_size=28,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 fc_out_channels=256,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 fc_norm_cfg=dict(type='BN',),
                 projector_cfg=dict(type='Linear'),
                 thred_u=0.1,
                 scale_u=1.0,
                 percent=0.3,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'

        super(ContrastiveHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
            None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        self.num_projectfc = num_projectfc
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)  # generate matrix
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fc_norm_cfg = fc_norm_cfg
        self.projector_cfg = projector_cfg
        self.fp16_enabled = False
        self.weight = 0.0  # init variable, this will be rewrite in different epoch
        self.thred_u = thred_u
        self.scale_u = scale_u
        self.percent = percent

        # build encoder module
        self.encoder = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.encoder.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            last_layer_dim = self.conv_out_channels

        # build projecter module
        self.projector = ModuleList()
        for j in range(self.num_projectfc - 1):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.projector.append(
                ConvModule(
                    fc_in_channels,
                    self.fc_out_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.fc_norm_cfg, )
            )
            last_layer_dim = self.fc_out_channels
        self.projector.append(
            build_linear_layer(
                self.projector_cfg,
                in_features=last_layer_dim,
                out_features=self.fc_out_channels)
        )

    def init_weights(self):
        super(ContrastiveHead, self).init_weights()

    def forward(self, x, cams, edges, masks, is_novel):
        # sample_sets = dict()
        #
        # # 1. get query and keys
        # if masks is not None:  # training phase
        #     sample_results = get_query_keys(cams, edges, masks, is_novel=is_novel, thred_u=self.thred_u,
        #                                     scale_u=self.scale_u, percent=self.percent)
        #     keeps_ = sample_results['keeps']
        #     keeps = keeps_.reshape(-1, 1, 1)
        #     keeps = keeps.expand(keeps.shape[0], x.shape[2],
        #                          x.shape[2])  # points of a reserved porposal are assigned 'keep'
        #     keeps_all = keeps.reshape(-1)
        # else:  # evaluation phase
        #     sample_results = get_query_keys_eval(cams)


        # 2. forward
        for conv in self.encoder:
            x = conv(x)
            print("endcoderx_shape",x.shape)
        x_pro = self.projector[0](x)
        print("self.projector[0](x)_shape", x_pro.shape)
        for i in range(1, len(self.projector) - 1):
            x_pro = self.projector[i](x_pro)
            print("self.projector[i]_shape", x_pro.shape)
        n, c, h, w = x_pro.shape
        x_pro = x_pro.permute(0, 2, 3, 1).reshape(-1, c)  # n,c,h,w -> n,h,w,c -> (nhw),c
        x_pro = self.projector[-1](x_pro)  # (nhw),c
        print("self.projector[-1]_shape", x_pro.shape)
        x_enhance = enhance_op(x)
        print("x_enhance_shape", x_enhance.shape)

        # 3. get vectors for queries and keys so that we can calculate contrastive loss
        if masks is not None:
            query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[1, 2])
            query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[1, 2])
            assert (0. not in query_pos_num) and (
                        0. not in query_neg_num), f"query should NOT be 0!!!! <-- contrastive_head.py"
            # keys
            sample_easy_pos = x_pro[keeps_all][sample_results['easy_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_easy_neg = x_pro[keeps_all][sample_results['easy_negative_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_pos = x_pro[keeps_all][sample_results['hard_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_neg = x_pro[keeps_all][sample_results['hard_negative_sets_N'].reshape(-1), :]  # *, 256
            # queries
            query_pos = (x_pro.reshape(-1, 28, 28, 256) * sample_results['query_pos_sets'].to(
                device=x_pro.device).unsqueeze(3)).sum(dim=[1, 2]) / query_pos_num.unsqueeze(
                1)  # foreground query for each proposal
            query_neg = (x_pro.reshape(-1, 28, 28, 256) * sample_results['query_neg_sets'].to(
                device=x_pro.device).unsqueeze(3)).sum(dim=[1, 2]) / query_neg_num.unsqueeze(1)

            # sample sets are used to calculate loss
            sample_sets['keeps_proposal'] = keeps_
            sample_sets['query_pos'] = query_pos[keeps_].unsqueeze(1)
            sample_sets['query_neg'] = query_neg[keeps_].unsqueeze(1)
            sample_sets['num_per_type'] = sample_results['num_per_type']
            sample_sets['sample_easy_pos'] = sample_easy_pos
            sample_sets['sample_easy_neg'] = sample_easy_neg
            sample_sets['sample_hard_pos'] = sample_hard_pos
            sample_sets['sample_hard_neg'] = sample_hard_neg

        return x_enhance, sample_sets

    def INFOloss(self, query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)

        query = torch.mean(query, dim=0).unsqueeze(0)  # mean op on all proposal, sharing-query
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]

        return batch_log

    def loss(self, easy_pos=None, easy_neg=None, hard_pos=None, hard_neg=None, query_pos=None, query_neg=None,
             t_easy=0.3, t_hard=0.7):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg: [B, 256]
        """
        alpha = 1.0

        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qpos_hard = self.INFOloss(query_pos, hard_pos, hard_neg, t_hard)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_Qneg_hard = self.INFOloss(query_neg, hard_neg, hard_pos, t_hard)
        loss_contrast = torch.mean(loss_Qpos_easy + loss_Qpos_hard + alpha * loss_Qneg_easy + alpha * loss_Qneg_hard)

        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')

        return loss_contrast * self.weight * 0.1

import torch.nn as nn
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class linear_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(linear_block, self).__init__()

        self.lnearconv = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.lnearconv(x)
        return x

def seg2edge(seg):
    bs = seg.size(0)
    segnp = seg.cpu().numpy()
    edge = np.zeros_like(segnp)
    for i in range(bs):
        edge[i] = vlseg2edge(segnp[i])
    return torch.tensor(edge).to(seg)

def mask2edge(seg):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=seg.device).reshape(1, 1, 3, 3).requires_grad_(False)
    #print("seg",torch.unique(seg))
    edge_targets = F.conv2d(seg, laplacian_kernel, padding=1)
    edge_targets = edge_targets.clamp(min=0)
    edge_targets[edge_targets > 0.1] = 1
    edge_targets[edge_targets <= 0.1] = 0
    return edge_targets

class ContrastiveHead_torch(nn.Module):

    def __init__(self,
                 num_convs=1,
                 num_projectfc=2,
                 in_channels=64,
                 conv_out_channels=64,
                 fc_out_channels=64,
                 thred_u=0.1,
                 scale_u=1.0,
                 percent=0.3):
        super(ContrastiveHead_torch, self).__init__()
        self.num_convs = num_convs
        self.num_projectfc = num_projectfc
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.fp16_enabled = False
        self.weight = 0.0  # init variable, this will be rewrite in different epoch
        self.thred_u = thred_u
        self.scale_u = scale_u
        self.percent = percent
        self.fake = True

        # build encoder module
        self.encoder = nn.ModuleList() #make a list to upconv
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            self.encoder.append(
                conv_block(in_channels,self.conv_out_channels))
            last_layer_dim = self.conv_out_channels

        # build projecter module
        self.projector = nn.ModuleList()
        for j in range(self.num_projectfc - 1):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.projector.append(
                conv_block(fc_in_channels,self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        self.projector.append(linear_block(in_ch=last_layer_dim, out_ch=self.fc_out_channels))

    def forward(self, x, masks,trained,faked):
        #mask for supervised  and prdict for unsupervised
        """
        We get average foreground pixel and background pixel for Quary pixel feature (by mask and thrshold for prdiction) 
        easy by bounary on the boundary and less than
        """
        self.fake = faked
        sample_sets = dict()
        if self.fake:
            #mask_squeeze = torch.squeeze(masks,dim=1)
            # print("mask_squeeze",masks.shape)
            edges = mask2edge(masks)
            # print("edges_shape",edges.shape)
            # print("edges_unique",torch.unique(edges))
        else:
            edges = None
        # 1. get query and keys
        if trained:  # training phase
            sample_results, flag = get_query_keys_sty(edges, masks, thred_u=self.thred_u, scale_u=self.scale_u, percent=self.percent, fake=self.fake)
            if flag==False:
                return x, sample_results, False
            keeps_ = sample_results['keeps']
            # print("keeps_",keeps_)#8
            # print("keeps_",keeps_.shape)
            # print("x",x.shape)#8 64 256 256
            keeps = keeps_.reshape(-1, 1, 1)
            # print("keeps",keeps.shape)#8 1 1
            keeps = keeps.expand(keeps.shape[0], x.shape[2], x.shape[3]) # points of a reserved porposal are assigned 'keep'
            # print("keeps_expand",keeps.shape)#8 256 256
            keeps_all = keeps.reshape(-1)
            # print("keep_all",keeps_all.shape)#524288s
        else:  # evaluation phase
            sample_results = get_query_keys_eval(masks)

        # 2. forward
        for conv in self.encoder:
            # print("x_shape",x.shape)#8 64 256 256
            x = conv(x)
        x_pro = self.projector[0](x)
        for i in range(1, len(self.projector) - 1):
            x_pro = self.projector[i](x_pro)
        # print("x_pro_init", x_pro.shape)#8 128 256 256
        n, c, h, w = x_pro.shape
        # print("x_pro_before", x_pro.shape)#8 128 256 256
        x_pro = x_pro.permute(0, 2, 3, 1).reshape(-1, c)  # n,c,h,w -> n,h,w,c -> (nhw),c
        # print("x_pro",x_pro.shape)#8 128 256 256
        x_pro = self.projector[-1](x_pro)  # (nhw),c
        # print("x_pro_after", x_pro.shape)#524288, 128
        #x_enhance = enhance_op(x)

        # 3. get vectors for queries and keys so that we can calculate contrastive loss
        if trained:
            query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
            query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
            # print("query_pos_num",query_pos_num)# 5
            # assert (0. not in query_pos_num) and (
            #             0. not in query_neg_num), f"query should NOT be 0!!!! <-- contrastive_head.py"
            # keys
            # print("x_pro_feature",x_pro.shape)#[524288 , 128]->8 sample
            # print("keep_all",keeps_all.shape)#8 256 256->524288
            # print("sample",sample_results['easy_positive_sets_N'].shape)#5,1,256,256
            # print("sample_results['easy_positive_sets_N'].reshape(-1)",sample_results['easy_positive_sets_N'].reshape(-1).shape)#327680 5 1 256 256/
            # print("x_pro[keeps_all]",x_pro[keeps_all].shape)#327680=256*256*5, 128. feature poins (N, 128)
            sample_easy_pos = x_pro[keeps_all][sample_results['easy_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_easy_neg = x_pro[keeps_all][sample_results['easy_negative_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_pos = x_pro[keeps_all][sample_results['hard_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_neg = x_pro[keeps_all][sample_results['hard_negative_sets_N'].reshape(-1), :]  # *, 256
            # queries
            # print("x_pro_newshape",x_pro.shape) #[524288, 128])
            # print("sample_results['query_pos_sets']",sample_results['query_pos_sets'].shape)# 5 1 256 256
            # print("sample_results['query_pos_sets']_unqiue",torch.unique(sample_results['query_pos_sets']))# 5 1 256 256
            # print("x_pro[keeps_all].reshape(-1, 256, 256, 128)",x_pro[keeps_all].reshape(-1, 256, 256, 64).shape)# 5 256 256 128
            # query_pos = (x_pro[keeps_all].reshape(-1, 256, 256, 128) * sample_results['query_pos_sets'].to(
            #     device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_pos_num.unsqueeze(
            #     1)  # foreground query for each proposal
            # query_neg = (x_pro[keeps_all].reshape(-1, 256, 256, 128) * sample_results['query_neg_sets'].to(
            #     device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_neg_num.unsqueeze(1)
            squeeze_sampletresult = sample_results['query_pos_sets'].squeeze(1)#5 256 256
            # print("squeeze_sampletresult",squeeze_sampletresult.shape)
            # print("squeeze_sampletresult",(x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_sampletresult.to(device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]).shape)#5 125
            query_pos = (x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_sampletresult.to(
                device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_pos_num
            #print("query_pos",query_pos)
            # if torch.any(torch.isnan(query_pos)):
            #     print("Warning!!!nan")
            squeeze_negsampletresult = sample_results['query_neg_sets'].squeeze(1)#5 256 256
            query_neg = (x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_negsampletresult.to(
                device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_neg_num
            # print("query_neg_num",query_neg_num.shape)#5 1
            # print("sample_results['query_pos_sets']",sample_results['query_pos_sets'].shape)# 5 1 256 256
            # print("query_neg",query_neg.shape)#5 128
            # print("query_pos", query_pos.shape)#5 128
            # print("Keep_", keeps_.shape)#5 128
            # sample sets are used to calculate loss
            sample_sets['keeps_proposal'] = keeps_
            sample_sets['query_pos'] = query_pos.unsqueeze(1)
            sample_sets['query_neg'] = query_neg.unsqueeze(1)
            sample_sets['num_per_type'] = sample_results['num_per_type']
            sample_sets['sample_easy_pos'] = sample_easy_pos
            sample_sets['sample_easy_neg'] = sample_easy_neg
            sample_sets['sample_hard_pos'] = sample_hard_pos
            sample_sets['sample_hard_neg'] = sample_hard_neg
            #print("sample_sets['query_pos']",sample_sets['query_pos'])
        return x, sample_sets, True

    def INFOloss(self, query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)

        query = torch.mean(query, dim=0).unsqueeze(0)  # mean op on all proposal, sharing-query
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]

        return batch_log

    def loss(self, easy_pos=None, easy_neg=None, hard_pos=None, hard_neg=None, query_pos=None, query_neg=None,
             t_easy=0.3, t_hard=0.7):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg: [B, 256]
        """
        alpha = 1.0

        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qpos_hard = self.INFOloss(query_pos, hard_pos, hard_neg, t_hard)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_Qneg_hard = self.INFOloss(query_neg, hard_neg, hard_pos, t_hard)
        loss_contrast = torch.mean(loss_Qpos_easy + loss_Qpos_hard + alpha * loss_Qneg_easy + alpha * loss_Qneg_hard)

        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')

        return loss_contrast * self.weight * 0.1

class ContrastiveHead_myself(nn.Module):

    def __init__(self,
                 num_convs=1,
                 num_projectfc=2,
                 in_channels=64,
                 conv_out_channels=64,
                 fc_out_channels=64,
                 thred_u=0.1,
                 scale_u=1.0,
                 percent=0.3):
        super(ContrastiveHead_myself, self).__init__()
        self.num_convs = num_convs #layer of encoder
        self.num_projectfc = num_projectfc #layers of projector
        self.in_channels = in_channels #channels number
        self.conv_out_channels = conv_out_channels# out put channels numbers
        self.fc_out_channels = fc_out_channels
        self.thred_u = thred_u
        self.scale_u = scale_u
        self.percent = percent
        self.fake = True

        # build encoder module
        self.encoder = nn.ModuleList() #make a list to upconv
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            self.encoder.append(
                conv_block(in_channels,self.conv_out_channels))
            last_layer_dim = self.conv_out_channels

        # build projecter module
        self.projector = nn.ModuleList()
        for j in range(self.num_projectfc - 1):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.projector.append(
                conv_block(fc_in_channels,self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        self.projector.append(linear_block(in_ch=last_layer_dim, out_ch=self.fc_out_channels))

    def forward(self, x, masks,trained,faked):
        #mask for supervised  and prdict for unsupervised
        """
        We get average foreground pixel and background pixel for Quary pixel feature (by mask and thrshold for prdiction)
        easy by bounary on the boundary and less than
        """
        self.fake = faked
        sample_sets = dict()
        if self.fake:
            edges = mask2edge(masks)
        else:
            edges = None
        # 1. get query and keys
        if trained:  # training phase
            sample_results, flag = get_query_keys_myself(edges, masks, thred_u=self.thred_u, scale_u=self.scale_u, percent=self.percent, fake=self.fake)
            if flag==False:
                return x, sample_results, flag
            keeps_ = sample_results['keeps']
            keeps = keeps_.reshape(-1, 1, 1)
            keeps = keeps.expand(keeps.shape[0], x.shape[2], x.shape[3]) # expand the flag(numbers of batch level) for the whole feature
            keeps_all = keeps.reshape(-1)
        else:  # evaluation phase
            sample_results = get_query_keys_eval(masks)

        # 2. forward
        for conv in self.encoder:
            x = conv(x)
        x_pro = self.projector[0](x)
        for i in range(1, len(self.projector) - 1):
            x_pro = self.projector[i](x_pro)
        n, c, h, w = x_pro.shape
        x_pro = x_pro.permute(0, 2, 3, 1).reshape(-1, c)  # n,c,h,w -> n,h,w,c -> (nhw),c
        x_pro = self.projector[-1](x_pro)  # (nhw),c

        # 3. get vectors for queries and keys so that we can calculate contrastive loss
        if trained:
            query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
            query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
            sample_easy_pos = x_pro[keeps_all][sample_results['easy_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_easy_neg = x_pro[keeps_all][sample_results['easy_negative_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_pos = x_pro[keeps_all][sample_results['hard_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_neg = x_pro[keeps_all][sample_results['hard_negative_sets_N'].reshape(-1), :]  # *, 256 choose the True postion feature as sample [oint
            squeeze_sampletresult = sample_results['query_pos_sets'].squeeze(1)#5 256 256 to get the whole pos map
            query_pos = (x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_sampletresult.to(
                device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_pos_num
            query_pos_set = x_pro[keeps_all][sample_results['query_pos_sets'].reshape(-1), :]
            squeeze_negsampletresult = sample_results['query_neg_sets'].squeeze(1)#5 256 256
            query_neg = (x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_negsampletresult.to(
                device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_neg_num
            query_neg_set = x_pro[keeps_all][sample_results['query_neg_sets'].reshape(-1), :]
            # sample sets are used to calculate loss
            sample_sets['keeps_proposal'] = keeps_
            sample_sets['query_pos'] = query_pos.unsqueeze(1)#N,HW,C 1,1,64
            sample_sets['query_neg'] = query_neg.unsqueeze(1)
            sample_sets['query_pos_set'] = query_pos_set #N,dims
            sample_sets['query_neg_set'] = query_neg_set #N,dims
            sample_sets['num_per_type'] = sample_results['num_per_type']
            sample_sets['sample_easy_pos'] = sample_easy_pos #N,64
            sample_sets['sample_easy_neg'] = sample_easy_neg
            sample_sets['sample_hard_pos'] = sample_hard_pos
            sample_sets['sample_hard_neg'] = sample_hard_neg
        return x, sample_sets, True


if __name__ == '__main__':
    con_head = ContrastiveHead_myself()
    mask = np.array(Image.open("/mnt/nas/sty/codes/Unsupervised/111.png").convert('L'))
    mask_tensor = np.expand_dims(mask,0)
    mask_tensor = np.expand_dims(mask_tensor,0)

    mask_torch = torch.tensor(mask_tensor)
    x = torch.randn((1, 64, 512, 512))
    y = con_head(x, mask_torch*1.0, True, True)

