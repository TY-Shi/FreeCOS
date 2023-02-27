r""" Evaluate mask prediction """
import torch
import numpy as np

from skimage import morphology
from torch.distributions.uniform import Uniform

#input NCHW


def noise_input(image_tensor,uniform_range=0.3):
    noise_uniform = Uniform(-uniform_range,uniform_range)
    noise_vector = noise_uniform.sample(image_tensor.shape).cuda(non_blocking=True)
    noise_img = image_tensor.mul(noise_vector) + image_tensor
    noise_img_normal = (noise_img-torch.min(noise_img))/(torch.max(noise_img)-torch.min(noise_img))
    return noise_img_normal


if __name__ == '__main__':
    x = torch.randn(2, 3, 128, 128)
    print(torch.min(x))
    x_normal = (x-torch.min(x))/(torch.max(x)-torch.min(x))
    print("x.shape[1:]",x.shape[1:])
    print("unique",torch.unique(x_normal))
    noise_x = noise_input(x_normal)
    print("x_noise",torch.unique(noise_x))