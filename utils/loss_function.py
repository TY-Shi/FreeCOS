import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class Contrastloss(nn.Module):
    def __init__(self):
        super(Contrastloss, self).__init__()
        self.epsilon = 1e-5

    def INFOloss(self,query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)
        #print("query", query)  # 1,1, 64
        query = torch.mean(query, dim=0).unsqueeze(0)  # mean op on all proposal, sharing-query
        #print("query",query.shape)#1,1, 64
        #print("query", query)  # 1,1, 64
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        #print("pos_sets",pos_sets.shape)#1 109458 64
        #print("neg_sets",neg_sets.shape)#1 4455 64
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        #print("Q_pos",Q_pos.shape)#1 109458
        #print("Q_neg",Q_neg.shape)#1 4455
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)#1
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        #print("single_in_log",single_in_log.shape)#1 109458
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]
        #print("batch_log",batch_log.shape)# 1
        #print("batch_log",batch_log)

        return batch_log

    def forward(self, easy_pos=None, easy_neg=None, hard_pos=None, hard_neg=None, query_pos=None, query_neg=None,
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
        # print("loss_Qpos_easy",torch.isnan(loss_Qpos_easy))
        # print("loss_Qpos_hard", torch.isnan(loss_Qpos_hard))
        # print("loss_Qneg_easy", torch.isnan(loss_Qneg_easy))
        # print("loss_Qneg_hard", torch.isnan(loss_Qneg_hard))
        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')
            return 0.0

        return loss_contrast * 0.1


class ContrastRegionloss(nn.Module):
    def __init__(self):
        super(ContrastRegionloss, self).__init__()
        self.epsilon = 1e-5

    def INFOloss(self,query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)
        #print("query", query)  # 1,1, 64
        #print("query_before", query.shape)#B, 1, 63
        query = query.unsqueeze(0)  # mean op on all proposal, sharing-query
        #print("query", query.shape)#1 ,1 ,64
        #print("pos_sets", pos_sets.shape)#N,64
        #print("query",query.shape)#1,1, 64
        #print("query", query)  # 1,1, 64
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        #print("pos_sets",pos_sets.shape)#1 109458 64
        #print("neg_sets",neg_sets.shape)#1 4455 64
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        #print("Q_pos",Q_pos.shape)#1 109458
        #print("Q_neg",Q_neg.shape)#1 4455
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)#1
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        #print("single_in_log",single_in_log.shape)#1 109458
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]
        #print("batch_log",batch_log.shape)# 1
        #print("batch_log",batch_log)

        return batch_log

    def forward(self, easy_pos=None, easy_neg=None, hard_pos=None, hard_neg=None, query_pos=None, query_neg=None,
                t_easy=0.3, t_hard=0.7):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg_set: [B, 1, 64]
            query_neg_set:[B, 1,64]
        """
        alpha = 1.0
        #print("loss_query_pos",query_pos.shape)#[1, 64]
        #print("loss_query_neg",query_neg.shape)#[1, 64]
        # print("loss_easy_neg",easy_neg.shape)#[N, 64]
        # print("loss_t_easy",easy_neg.shape)#[N, 64]

        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qpos_hard = self.INFOloss(query_pos, hard_pos, hard_neg, t_hard)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_Qneg_hard = self.INFOloss(query_neg, hard_neg, hard_pos, t_hard)
        loss_contrast = torch.mean(loss_Qpos_easy + loss_Qpos_hard + alpha * loss_Qneg_easy + alpha * loss_Qneg_hard)
        # print("loss_Qpos_easy",torch.isnan(loss_Qpos_easy))
        # print("loss_Qpos_hard", torch.isnan(loss_Qpos_hard))
        # print("loss_Qneg_easy", torch.isnan(loss_Qneg_easy))
        # print("loss_Qneg_hard", torch.isnan(loss_Qneg_hard))
        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')
            return 0.0

        return loss_contrast * 0.1


class ContrastRegionloss_noedge(nn.Module):
    def __init__(self):
        super(ContrastRegionloss_noedge, self).__init__()
        self.epsilon = 1e-5

    def INFOloss(self,query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)
        #print("query", query)  # 1,1, 64
        #print("query_before", query.shape)#B, 1, 63
        query = query.unsqueeze(0)  # mean op on all proposal, sharing-query
        #print("query", query.shape)#1 ,1 ,64
        #print("pos_sets", pos_sets.shape)#N,64
        #print("query",query.shape)#1,1, 64
        #print("query", query)  # 1,1, 64
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        #print("pos_sets",pos_sets.shape)#1 109458 64
        #print("neg_sets",neg_sets.shape)#1 4455 64
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        #print("Q_pos",Q_pos.shape)#1 109458
        #print("Q_neg",Q_neg.shape)#1 4455
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)#1
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        #print("single_in_log",single_in_log.shape)#1 109458
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]

        return batch_log

    def forward(self, easy_pos=None, easy_neg=None, query_pos=None, query_neg=None,
                t_easy=0.3):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg_set: [B, 1, 64]
            query_neg_set:[B, 1,64]
        """
        alpha = 1.0
        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_contrast = torch.mean(loss_Qpos_easy  + alpha * loss_Qneg_easy)

        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')
            return 0.0

        return loss_contrast * 0.1


class ContrastRegionloss_supunsup(nn.Module):
    def __init__(self):
        super(ContrastRegionloss_supunsup, self).__init__()
        self.epsilon = 1e-5

    def INFOloss(self,query, pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)
        #print("query", query)  # 1,1, 64
        query = torch.mean(query, dim=0).unsqueeze(0)  # mean op on all proposal, sharing-query
        #print("query",query.shape)#1,1, 64
        #print("query", query)  # 1,1, 64
        pos_sets = pos_sets.unsqueeze(0) - query
        neg_sets = neg_sets.unsqueeze(0) - query
        #print("pos_sets",pos_sets.shape)#1 109458 64
        #print("neg_sets",neg_sets.shape)#1 4455 64
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        #print("Q_pos",Q_pos.shape)#1 109458
        #print("Q_neg",Q_neg.shape)#1 4455
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)#1
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        #print("single_in_log",single_in_log.shape)#1 109458
        batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]
        #print("batch_log",batch_log.shape)# 1
        #print("batch_log",batch_log)

        return batch_log

    def forward(self, easy_pos=None, easy_neg=None, query_pos=None, query_neg=None,
                t_easy=0.3):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg_set: [B, 1, 64]
            query_neg_set:[B, 1,64]
        """
        alpha = 1.0
        #print("loss_query_pos",query_pos.shape)#[1, 64]
        #print("loss_query_neg",query_neg.shape)#[1, 64]
        # print("loss_easy_neg",easy_neg.shape)#[N, 64]
        # print("loss_t_easy",easy_neg.shape)#[N, 64]

        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_contrast = torch.mean(loss_Qpos_easy + alpha * loss_Qneg_easy)
        # print("loss_Qpos_easy",torch.isnan(loss_Qpos_easy))
        # print("loss_Qpos_hard", torch.isnan(loss_Qpos_hard))
        # print("loss_Qneg_easy", torch.isnan(loss_Qneg_easy))
        # print("loss_Qneg_hard", torch.isnan(loss_Qneg_hard))
        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')
            return 0.0

        return loss_contrast * 0.1


class Symmetruc_MSE(nn.Module):
    def __init__(self):
        super(Symmetruc_MSE, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        score = torch.mean((predict-target)**2)

        return score


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Change the data dimsion

    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        # N_q , 1

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)
            # [N_q N_neg]

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        # print("labels",labels.shape)
        # print("logits",logits.shape)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

# def INFOloss(query, pos_sets, neg_sets, tem):
#     ''' Dense INFOloss (pixel-wise)
#     example:
#         query: 5x1x256  5 is the number of proposals for a batch
#         pos_sets: 135x256
#         neg_sets: 170x256
#     '''
#     N = pos_sets.shape[0]
#     N = max(1, N)
#     query = torch.mean(query, dim=0).unsqueeze(0) # mean op on all proposal, sharing-query
#     #print("query",query.shape)#1,1, 64
#     #print("pos_sets",pos_sets.shape)#1 120 64
#     #print("neg_sets",neg_sets.shape)#1 200 64
#     query = F.normalize(query, dim=-1)
#     pos_sets = F.normalize(pos_sets, dim=-1)
#     neg_sets = F.normalize(neg_sets, dim=-1)
#     Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
#     Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
#     #print("Q_pos",Q_pos.shape)#1 120
#     #print("Q_neg",Q_neg.shape)#1 200
#     Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
#     #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)
#     Q_neg_pos1_expsum= Q_neg_exp_sum + torch.exp(Q_pos / tem)
#     #print("Q_neg_pos1_expsum",Q_neg_pos1_expsum.shape) [1,120]
#     Q_pos_exp_list =  torch.exp(Q_pos / tem)
#     #print("Q_pos_exp_list",Q_pos_exp_list.shape)#[1,120]
#     single_nce = -1 * torch.log(Q_pos_exp_list / Q_neg_pos1_expsum)
#     #print("single_nce",single_nce.shape)
#     batch_log =  torch.mean(single_nce, dim=1)
#     #print("batch_log",batch_log.shape)# 1
#     return batch_log*0.1


class ContrastRegionloss_repeatNCE(nn.Module):
    def __init__(self):
        super(ContrastRegionloss_repeatNCE, self).__init__()
        self.infonceloss = InfoNCE()

    def forward(self, pos_feature, neg_feature):
        return self.infonceloss(pos_feature, pos_feature, neg_feature)
        # return self.sum_nce / pos_num

class ContrastRegionloss_quaryrepeatNCE(nn.Module):
    def __init__(self):
        super(ContrastRegionloss_quaryrepeatNCE, self).__init__()
        self.infonceloss = InfoNCE()

    def forward(self, quary, pos_feature, neg_feature):
        if len(quary)>=len(pos_feature):
            quary = quary[:len(pos_feature),:]
        else:
            pos_feature = pos_feature[:len(quary),:]
        return self.infonceloss(quary, pos_feature, neg_feature)


class ContrastRegionloss_NCE(nn.Module):
    def __init__(self):
        super(ContrastRegionloss_NCE, self).__init__()
        self.epsilon = 1e-5

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
        # print("query",query.shape)#1,1, 64
        # print("pos_sets",pos_sets.shape)#1 120 64
        # print("neg_sets",neg_sets.shape)#1 200 64
        query = F.normalize(query, dim=-1)
        pos_sets = F.normalize(pos_sets, dim=-1).unsqueeze(0)
        neg_sets = F.normalize(neg_sets, dim=-1).unsqueeze(0)
        # print("query",query.shape)
        # print("pos_sets",pos_sets.shape)
        # print("neg_sets",neg_sets.shape)
        Q_pos = F.cosine_similarity(query, pos_sets, dim=2)  # [1, 135]
        Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
        # print("Q_pos",Q_pos.shape)#1 120
        # print("Q_neg",Q_neg.shape)#1 200
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
        # print("Q_neg_exp_sum",Q_neg_exp_sum.shape)
        Q_neg_pos1_expsum = Q_neg_exp_sum + torch.exp(Q_pos / tem)
        # print("Q_neg_pos1_expsum",Q_neg_pos1_expsum.shape) [1,120]
        Q_pos_exp_list = torch.exp(Q_pos / tem)
        # print("Q_pos_exp_list",Q_pos_exp_list.shape)#[1,120]
        single_nce = -1 * torch.log(Q_pos_exp_list / Q_neg_pos1_expsum)
        # print("single_nce",single_nce.shape)
        batch_log = torch.mean(single_nce, dim=1)
        # print("batch_log",batch_log.shape)# 1
        return batch_log

    def forward(self, easy_pos=None, easy_neg=None, query_pos=None, query_neg=None,
                t_easy=0.07):
        """
            easy_pos: [B, N, 256]
            easy_neg: [B, N, 256]
            hard_pos: [B, N, 256]
            hard_neg: [B, N, 256]
            pos_center: [B, 1, 256]
            query_pos: [B, 256]
            query_neg_set: [B, 1, 64]
            query_neg_set:[B, 1,64]
        """
        alpha = 1.0
        #print("loss_query_pos",query_pos.shape)#[1, 64]
        #print("loss_query_neg",query_neg.shape)#[1, 64]
        # print("loss_easy_neg",easy_neg.shape)#[N, 64]
        # print("loss_t_easy",easy_neg.shape)#[N, 64]

        loss_Qpos_easy = self.INFOloss(query_pos, easy_pos, easy_neg, t_easy)
        loss_Qneg_easy = self.INFOloss(query_neg, easy_neg, easy_pos, t_easy)
        loss_contrast = torch.mean(loss_Qpos_easy + alpha * loss_Qneg_easy)
        # print("loss_Qpos_easy",torch.isnan(loss_Qpos_easy))
        # print("loss_Qpos_hard", torch.isnan(loss_Qpos_hard))
        # print("loss_Qneg_easy", torch.isnan(loss_Qneg_easy))
        # print("loss_Qneg_hard", torch.isnan(loss_Qneg_hard))
        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')
            return 0.0

        return loss_contrast * 0.1


class ContrastRegionloss_AllNCE(nn.Module):
    def __init__(self):
        super(ContrastRegionloss_AllNCE, self).__init__()
        self.epsilon = 1e-5
        self.queue_len = 2000

    def INFONCEloss(self, pos_sets, compare_pos_set, neg_sets, tem=0.07):
        ''' Dense INFOloss (pixel-wise)
        example:
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)
        pos_sets = F.normalize(pos_sets, dim=-1)
        neg_sets = F.normalize(neg_sets, dim=-1)
        com_pos_set = F.normalize(compare_pos_set, dim=-1)
        #print("pos_sets",pos_sets.shape)
        #print("neg_sets",neg_sets.shape)
        #print("com_pos_set",com_pos_set.shape)
        Q_pos = torch.div(torch.matmul(pos_sets, com_pos_set.T), tem)
        #print("Q_pos",torch.min(Q_pos))
        Q_neg = torch.div(torch.matmul(pos_sets, neg_sets.T), tem)
        #print("Q_neg",torch.min(Q_neg))
        Q_pos_exp_list = torch.exp(Q_pos)
        Q_neg_exp_list = torch.exp(Q_neg)
        Q_neg_exp_sum = torch.sum(Q_neg_exp_list, dim=1)
        #print("Q_pos_exp_list",Q_pos_exp_list.shape)
        #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)
        Q_neg_exp_sum_repeat = Q_neg_exp_sum.unsqueeze(1).repeat(1,Q_pos_exp_list.shape[1])
        #print("Q_neg_exp_sum_repeat",Q_neg_exp_sum_repeat.shape)
        matrix_log = -1 * torch.log(Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat+ 1e-8))
        print("matrix_log",matrix_log.shape)
        #print("(Q_pos_exp_list + Q_neg_exp_sum_repeat)_min",torch.min(Q_pos_exp_list + Q_neg_exp_sum_repeat))
        #print("Q_pos_exp_list_min", torch.min(Q_pos_exp_list))
        #print("Q_neg_exp_sum_repeat_min",torch.min(Q_neg_exp_sum_repeat))
        #print("Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat)_min", torch.min(Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat + 1e-8)))

        #print("(Q_pos_exp_list + Q_neg_exp_sum_repeat)",torch.isnan(Q_pos_exp_list + Q_neg_exp_sum_repeat).int().sum())
        #print("Q_pos_exp_list", torch.isnan(Q_pos_exp_list).int().sum())
        #print("Q_neg_exp_sum_repeat",torch.isnan(Q_neg_exp_sum_repeat).int().sum())
        #print("Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat)", torch.isnan(Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat + 1e-8)).int().sum())
        #print("matrix_log",matrix_log.shape)
        #Ones_tril = torch.tril(torch.ones(matrix_log.shape[0], matrix_log.shape[1])).cuda()
        #Up_torch = Ones_tril - torch.eye(matrix_log.shape[0], matrix_log.shape[1]).cuda() * matrix_log
        #batch_sum = torch.sum(Ones_tril) / (matrix_log.shape[0] * matrix_log.shape[0] * 0.5 - matrix_log.shape[0])
        batch_sum = torch.mean(matrix_log)
        print("batch_sum",batch_sum)
        return batch_sum


    def INFONCEloss_single(self, pos_sets, neg_sets, tem=0.07):
        ''' Dense INFOloss (pixel-wise)
        example:
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        N = pos_sets.shape[0]
        N = max(1, N)
        pos_sets = F.normalize(pos_sets, dim=-1)
        neg_sets = F.normalize(neg_sets, dim=-1)

        Q_pos = torch.div(torch.matmul(pos_sets, pos_sets.T), tem)
        #print("Q_pos",torch.min(Q_pos))
        Q_neg = torch.div(torch.matmul(pos_sets, neg_sets.T), tem)
        #print("Q_neg",torch.min(Q_neg))
        Q_pos_exp_list = torch.exp(Q_pos)
        Q_neg_exp_list = torch.exp(Q_neg)
        Q_neg_exp_sum = torch.sum(Q_neg_exp_list, dim=1)
        Q_pos_exp_sum = torch.sum(Q_pos_exp_list, dim=1)
        #print("Q_pos_exp_list",Q_pos_exp_list.shape)
        #print("Q_neg_exp_sum",Q_neg_exp_sum.shape)
        #Q_neg_exp_sum_repeat = Q_neg_exp_sum.unsqueeze(1).repeat(1,Q_pos_exp_list.shape[1])
        #print("Q_neg_exp_sum_repeat",Q_neg_exp_sum_repeat.shape)
        #matrix_log = -1 * torch.log(Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat+ 1e-8))
        matrix_log = -1 * torch.log(Q_pos_exp_sum / (Q_pos_exp_sum + Q_neg_exp_sum))
        # Ones_tril = torch.tril(torch.ones(matrix_log.shape[0], matrix_log.shape[1])).cuda()
        # Up_torch = Ones_tril - torch.eye(matrix_log.shape[0], matrix_log.shape[1]).cuda() * matrix_log
        batch_sum = torch.mean(matrix_log)
        #print("batch_sum",batch_sum)
        return batch_sum

    def INFOloss_similarity(self,pos_sets, neg_sets, tem):
        ''' Dense INFOloss (pixel-wise)
        example:
            query: 5x1x256  5 is the number of proposals for a batch
            pos_sets: 135x256
            neg_sets: 170x256
        '''
        pos_sets = F.normalize(pos_sets, dim=-1)
        neg_sets = F.normalize(neg_sets, dim=-1)

        Q_pos = torch.div(torch.matmul(pos_sets, pos_sets.T), tem)
        Q_neg = torch.div(torch.matmul(pos_sets, neg_sets.T), tem)
        #print("pos_sets",pos_sets.shape)#1 109458 64
        #print("neg_sets",neg_sets.shape)#1 4455 64
        #print("Q_pos",Q_pos.shape)#1 109458
        #print("Q_neg",Q_neg.shape)#1 4455
        Q_pos_exp_list = torch.exp(Q_pos)
        Q_neg_exp_list = torch.exp(Q_neg)
        Q_neg_exp_sum = torch.sum(Q_neg_exp_list, dim=1)  # [1]
        #Q_pos_exp_sum = torch.sum(torch.exp(Q_pos), dim=1)  # [1]
        Q_neg_exp_sum_repeat = Q_neg_exp_sum.unsqueeze(1).repeat(1, Q_pos_exp_list.shape[1])
        #single_in_log = torch.exp(Q_pos) / (torch.exp(Q_pos) + Q_neg_exp_sum.unsqueeze(1))  # [1, 135]
        matrix_log = -1 * torch.log(Q_pos_exp_list / (Q_pos_exp_list + Q_neg_exp_sum_repeat + 1e-8))
        #print("single_in_log",single_in_log.shape)#1 109458
        #batch_log = torch.sum(-1 * torch.log(single_in_log), dim=1) / N  # [1]
        batch_log = torch.sum(torch.log(matrix_log)) /(matrix_log.shape[0]*matrix_log.shape[1])  # [1]
        #print("batch_log",batch_log)
        return batch_log

    #sample_set_sup['sample_easy_pos'], sample_set_sup['sample_easy_neg'], sample_set_unsup['sample_easy_pos'], sample_set_unsup['sample_easy_neg'])
    def forward(self, pos_feature, neg_feature,t_easy=0.07):
        """
            easy_pos: [N, 256]
            easy_neg: [N, 256]
            hard_pos: [N, 256]
            hard_neg: [N, 256]
        """
        alpha = 1.0
        #print("loss_query_pos",query_pos.shape)#[1, 64]
        #print("loss_query_neg",query_neg.shape)#[1, 64]
        # print("loss_easy_neg",easy_neg.shape)#[N, 64]
        # print("loss_t_easy",easy_neg.shape)#[N, 64]
        # print("unlabel_pos",unlabel_pos.shape)
        # print("self.queue_len//2",self.queue_len//2)
        # print("unlabel_pos.shape[0]",unlabel_pos.shape[0])
        # print("range",min(self.queue_len//2, np.array(unlabel_pos.shape[0])))
        # pos_feature = torch.cat((unlabel_pos[:min(self.queue_len//2, unlabel_pos.shape[0]),:], label_pos[:min(self.queue_len//2, label_pos.shape[0]),:]), dim=0)
        # neg_feature = torch.cat((unlabel_neg[:min(self.queue_len//2, unlabel_neg.shape[0]),:], label_neg[:min(self.queue_len//2, label_neg.shape[0]),:]), dim=0)
        # print("pos_feature",pos_feature.shape)
        # print("neg_feature",neg_feature.shape)
        loss_labelpos_labelneg = self.INFONCEloss_single(pos_feature, neg_feature, t_easy)
        #loss_labelpos_labelneg = self.INFOloss_similarity(pos_feature, neg_feature, t_easy)
        # loss_labelpos_labelneg = self.INFONCEloss(label_pos, label_pos, label_neg, t_easy)
        # loss_labelposcompare_labelneg = self.INFONCEloss(label_pos, unlabel_pos, label_neg, t_easy)
        #
        # loss_labelpos_unlabelneg = self.INFONCEloss(label_pos, label_pos, unlabel_neg, t_easy)
        # loss_labelposcompare_unlabelneg = self.INFONCEloss(label_pos, unlabel_pos, unlabel_neg, t_easy)
        # #
        # loss_unlabelpos_labelneg = self.INFONCEloss(unlabel_pos, unlabel_pos, label_neg, t_easy)
        # loss_unlabelposcompare_labelneg = self.INFONCEloss(unlabel_pos, label_pos, label_neg, t_easy)
        # #
        # loss_unlabelpos_unlabelneg = self.INFONCEloss(unlabel_pos, unlabel_pos, unlabel_neg, t_easy)
        # loss_unlabelposcompare_unlabelneg = self.INFONCEloss(unlabel_pos, label_pos, unlabel_neg, t_easy)

        #loss_contrast = torch.mean(loss_labelpos_labelneg + loss_labelpos_unlabelneg + loss_unlabelpos_labelneg + loss_unlabelpos_unlabelneg + loss_unlabelposcompare_unlabelneg + loss_unlabelposcompare_labelneg + loss_labelposcompare_unlabelneg + loss_labelposcompare_labelneg)
        loss_contrast = loss_labelpos_labelneg
        # check NaN
        if True in torch.isnan(loss_contrast):
            print('NaN occurs in contrastive_head loss')
            return 0.0

        return loss_contrast * 0.1

def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target, **kwargs):
        return TripletSemiHardLoss(target, input, self.device)

# class Triplet(nn.Module):
#     def __init__(self):
#         super(Triplet, self).__init__()
#         self.margin = 0.2
#
#     def forward(self, anchor, positive, negative):
#         pos_dist = (anchor - positive).pow(2).sum(1)
#         neg_dist = (anchor - negative).pow(2).sum(1)
#         loss = F.relu(pos_dist - neg_dist + self.margin)
#         return loss.mean()

class Triplet(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(Triplet, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        min_len = min(len(anchor), len(pos), len(neg))
        anchor_cut = anchor[:min_len,:]
        pos_cut = pos[:min_len,:]
        neg_cut = neg[:min_len,:]
        if self.margin is None:
            num_samples = anchor_cut.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor_cut.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor_cut-pos_cut, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor_cut-neg_cut, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor_cut, pos_cut, neg_cut)

        return loss


if __name__ == '__main__':
    # query = torch.zeros((1, 1, 64))
    # pos_set = torch.randn((1, 120,64))
    # neg_set = torch.randn((1, 200,64))
    # batch_loss = INFOloss(query, pos_set, neg_set, tem=0.3)

    # pos_sets = torch.randn((120,64))
    # neg_sets = torch.randn((200,64))
    #
    # N = pos_sets.shape[0]
    # N = max(1, N)
    # pos_sets = F.normalize(pos_sets, dim=-1)
    # neg_sets = F.normalize(neg_sets, dim=-1)
    # print("pos_sets", pos_sets.shape)
    # print("neg_sets", neg_sets.shape)
    # Q_pos = torch.div(torch.matmul(pos_sets, pos_sets.T),0.07)
    # Q_neg = torch.div(torch.matmul(pos_sets, neg_sets.T),0.07)
    # print("Q_pos",Q_pos.shape)
    # print("Q_neg",Q_neg.shape)
    # Q_pos_exp_list = torch.exp(Q_pos)
    # Q_neg_exp_list = torch.exp(Q_neg)
    # Q_neg_exp_sum = torch.sum(Q_neg_exp_list, dim=1)
    # print("Q_neg_exp_sum",Q_neg_exp_sum.shape)
    # matrix_log = -1 * torch.log(Q_pos_exp_list/(Q_pos_exp_list+Q_neg_exp_sum))
    # print("matrix_log",matrix_log.shape)
    # Ones_tril = torch.tril(torch.ones(matrix_log.shape[0],matrix_log.shape[1]))
    # Up_torch = Ones_tril - torch.eye(matrix_log.shape[0],matrix_log.shape[1]) * matrix_log
    # print("Ones_tril",Up_torch.shape)
    # batch_sum = torch.sum(Up_torch) / (matrix_log.shape[0] * matrix_log.shape[0]* 0.5 -matrix_log.shape[0])
    # print("batch_sum",batch_sum)

    # loss = InfoNCE()
    # batch_size, num_negative, embedding_size = 32, 48, 128
    # query = torch.randn(batch_size, embedding_size)
    # positive_key = torch.randn(batch_size, embedding_size)
    # negative_keys = torch.zeros((num_negative, embedding_size))
    # output = loss(query, positive_key, query)
    # print("output",output)

    pos_sets = torch.randn((120, 64))
    pos_sets_1 = None
    neg_feature = torch.cat((pos_sets,pos_sets),dim=0)
    print("neg_feature",neg_feature.shape)
    #
    # positive_key_unlabel = torch.ones((batch_size, embedding_size))
    # negative_keys_unlabel = torch.zeros((num_negative, embedding_size))
    # loss_cet = ContrastRegionloss_AllNCE()
    # loss_contrast = loss_cet(positive_key.cuda(), negative_keys.cuda(), positive_key_unlabel.cuda(), negative_keys_unlabel.cuda())
    # print("loss_contrast",loss_contrast)
    #
    # target = torch.randint(5, (5,), dtype=torch.int64)
    # print("target = torch.randint(5, (3,), dtype=torch.int64)", target)
    # Q_neg = F.cosine_similarity(query, neg_sets, dim=2)  # [1, 170]
    # # print("Q_pos",Q_pos.shape)#1 120
    # # print("Q_neg",Q_neg.shape)#1 200
    # Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem), dim=1)  # [1]
    # # print("Q_neg_exp_sum",Q_neg_exp_sum.shape)
    # Q_neg_pos1_expsum = Q_neg_exp_sum + torch.exp(Q_pos / tem)
    # # print("Q_neg_pos1_expsum",Q_neg_pos1_expsum.shape) [1,120]
    # Q_pos_exp_list = torch.exp(Q_pos / tem)
    # # print("Q_pos_exp_list",Q_pos_exp_list.shape)#[1,120]
    # single_nce = -1 * torch.log(Q_pos_exp_list / Q_neg_pos1_expsum)
    # # print("single_nce",single_nce.shape)
    # batch_log = torch.mean(single_nce, dim=1)
    # # print("batch_log",batch_log.shape)# 1