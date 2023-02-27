import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np

def eightway_affinity_kld(probs, size=1):
    b, c, h, w = probs.size()
    if probs.dim() != 4:
        raise Exception('Only support for 4-D tensors!')
    p = size
    probs_pad = F.pad(probs, [p]*4, mode='replicate')
    bot_epsilon = 1e-4
    top_epsilon = 1.0
    neg_probs_clamp = torch.clamp(1.0 - probs, bot_epsilon, top_epsilon)
    probs_clamp = torch.clamp(probs, bot_epsilon, top_epsilon)
    kldiv_groups = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if st_y == size and st_x == size:
                # Ignore the center pixel/feature.
                continue
            probs_paired = probs_pad[:, :, st_y:st_y+h, st_x:st_x+w] * probs
            neg_probs_paired = torch.clamp(
                1.0-probs_paired, bot_epsilon, top_epsilon)
            probs_paired = torch.clamp(probs_paired, bot_epsilon, top_epsilon)
            kldiv = probs_paired * torch.log(probs_paired/probs_clamp) \
                + neg_probs_paired * \
                torch.log(neg_probs_paired/neg_probs_clamp)
            kldiv_groups.append(kldiv)
    return torch.cat(kldiv_groups, dim=1)

def fourway_affinity_kld(probs, size=1):
    b, c, h, w = probs.size()
    if probs.dim() != 4:
        raise Exception('Only support for 4-D tensors!')
    p = size
    probs_pad = F.pad(probs, [p]*4, mode='replicate')
    bot_epsilon = 1e-4
    top_epsilon = 1.0
    neg_probs_clamp = torch.clamp(1.0 - probs, bot_epsilon, top_epsilon)
    probs_clamp = torch.clamp(probs, bot_epsilon, top_epsilon)
    kldiv_groups = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if abs(st_y - st_x) == size:
                probs_paired = probs_pad[:, :,
                                         st_y:st_y+h, st_x:st_x+w] * probs
                neg_probs_paired = torch.clamp(
                    1.0-probs_paired, bot_epsilon, top_epsilon)
                probs_paired = torch.clamp(
                    probs_paired, bot_epsilon, top_epsilon)
                kldiv = probs_paired * torch.log(probs_paired/probs_clamp) \
                    + neg_probs_paired * \
                    torch.log(neg_probs_paired/neg_probs_clamp)
                kldiv_groups.append(kldiv)
    return torch.cat(kldiv_groups, dim=1)


def colorize_mask(mask):
    # mask: numpy array of the mask
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

if __name__ == '__main__':
    input = torch.randn(2, 1, 512, 512)
    output = eightway_affinity_kld(input, size=3)
    print("outshape",output.shape)
    print("output_unique",torch.unique(output))
