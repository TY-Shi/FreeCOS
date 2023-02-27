# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:elastic_transform.py
@time:2018/10/1221:56
"""
# Import stuff
import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import PIL.Image as Image

# Function to distort image  alpha = im_merge.shape[1]*2、sigma=im_merge.shape[1]*0.08、alpha_affine=sigma
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
    # 其中center_square是图像的中心，square_size=512//3=170
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
    # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
    # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
    # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
    # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    #x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))  # 网格采样点函数
    #indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


if __name__ == '__main__':
    mask = Image.open(os.path.join('/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_gtvessel_width', '81.png')).convert('L')
    image = Image.open(os.path.join('/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_grayvessel_width', '81.png')).convert('L')
    #image_elastic_PIL.save('./img_elastic.png')
    image_array = np.array(image)
    mask_array = np.array(mask)
    displacement = np.random.randn(2, 7, 7) * 25
    #image_deformed = elasticdeform.deform_random_grid(image_array, sigma=25, points=3)
    #mask_deformed = elasticdeform.deform_random_grid(mask_array, sigma=25, points=3)
    image_deformed = elastic_transform(image_array,image_array.shape[1]*2, image_array.shape[1]*0.08, image_array.shape[1]*2)
    mask_deformed = elastic_transform(mask_array,image_array.shape[1]*2, image_array.shape[1]*0.08, image_array.shape[1]*2)
    print("unique:",np.unique(mask_deformed))
    #deform_grid(X, displacement, order=3, mode='constant', cval=0.0, crop=None, prefilter=True, axis=None, affine=None, rotate=None, zoom=None)
    #image_elastic, gt_elastic = elastic_transform_PIL(image, mask, image_array.shape[1]*2, image_array.shape[1]*0.08, image_array.shape[1]*0.08)
    image_elastic_PIL = Image.fromarray((image_deformed).astype('uint8')).convert('L')
    mask_elastic_PIL = Image.fromarray((mask_deformed).astype('uint8')).convert('L')
    #gt_elastic_PIL = Image.fromarray((gt_elastic).astype('uint8')).convert('L')
    image_elastic_PIL.save('./img_elastic.png')
    mask_elastic_PIL.save('./mask_elastic.png')
    #gt_elastic_PIL.save('./gt_elastic.png')