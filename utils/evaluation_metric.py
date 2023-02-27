r""" Evaluate mask prediction """
import torch
import numpy as np

from skimage import morphology
from sklearn.metrics import roc_auc_score
def calAUC(pred,gt):
    #get 1-D
    pred_1D = np.array(pred.cpu()).flatten()
    gt_1D = np.array(gt.cpu()).flatten()
    # print("gt_1D",gt_1D.shape)
    # print("pred_1D",pred_1D.shape)
    # print("gt_unqiue",np.unique(gt_1D))
    # print("pred_1D_unqiue",np.unique(pred_1D))
    AUC = roc_auc_score(gt_1D, pred_1D)
    # N = 0
    # P = 0
    # neg_prob = []
    # pos_prob = []
    # for key, value in enumerate(gt_1D):
    #     if(value==1):
    #         P += 1
    #         pos_prob.append(pred_1D[key])
    #     else:
    #         N +=1
    #         neg_prob.append(pred_1D[key])
    # number = 0
    # for pos in pos_prob:
    #     for neg in neg_prob:
    #         if(pos>neg):
    #             number +=1
    #         elif (pos==neg):
    #             number += 0.5
    return AUC

def computeF1(pred, gt):
    """

    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score * 100, precision * 100, recall * 100

def jc(pred, gt):
    """
    Jaccard coefficient
    Computes the Jaccard coefficient between the binary objects in two images.
    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = pred.cpu().numpy()
    reference = gt.cpu().numpy()
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc

def compute_allXCAD(pred, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    Acc = (tp+tn)/(tp+fn+tn+fp)
    Sp = tn/(tn+fp+ epsilon)
    jc_score = jc(pred,gt)

    return f1_score * 100, precision * 100, recall * 100, Sp * 100, Acc * 100, jc_score *100

def compute_allRetinal(pred, pred_con, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7
    # print("fp",fp)
    # print("fn",fn)
    # print("tp",tp)
    # print("tp",tn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    Acc = (tp+tn)/(tp+fn+tn+fp)
    Sp = tn/(tn+fp+ epsilon)
    jc_score = jc(pred,gt)
    AUC = calAUC(pred_con, gt)
    return f1_score * 100, precision * 100, recall * 100, Sp * 100, Acc * 100, jc_score *100, AUC


if __name__ == '__main__':
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    pred = np.array([0.1, 0.4, 0.5, 0.6, 0.35, 0.8, 0.9, 0.4, 0.25, 0.5])
    print("sklearn auc:", roc_auc_score(y,pred))
    print("my auc calc by area:", calAUC(pred,y))
    print("my auc calc by prob:", calAUC(pred,y))