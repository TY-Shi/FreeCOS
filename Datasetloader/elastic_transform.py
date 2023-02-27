import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import PIL.Image as Image
import os
import elasticdeform

def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=0, mode='constant'.reshape(shape))
    return imageC

def elastic_transform_PIL(image, gt, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    image_array = np.array(image)#H*W
    gt_array = np.array(gt)#H*W
    shape = image_array.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image_array, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101,flags=cv2.INTER_NEAREST)
    image_gt = cv2.warpAffine(gt_array, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101,flags=cv2.INTER_NEAREST)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    image_elastic = map_coordinates(imageB, indices, order=0, mode='constant').reshape(shape)
    gt_elastic = map_coordinates(image_gt, indices, order=0, mode='constant').reshape(shape)

    return image_elastic, gt_elastic



if __name__ == '__main__':
    mask = Image.open(os.path.join('/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_gtvessel_width', '81.png')).convert('L')
    image = Image.open(os.path.join('/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_grayvessel_width', '81.png')).convert('L')
    #image_elastic_PIL.save('./img_elastic.png')
    image_array = np.array(image)
    mask_array = np.array(mask)
    displacement = np.random.randn(2, 7, 7) * 25
    #image_deformed = elasticdeform.deform_random_grid(image_array, sigma=25, points=3)
    #mask_deformed = elasticdeform.deform_random_grid(mask_array, sigma=25, points=3)
    # image_deformed = elasticdeform.deform_grid(image_array,displacement,mode='nearest')
    # mask_deformed = elasticdeform.deform_grid(mask_array, displacement,mode='nearest')
    #deform_grid(X, displacement, order=3, mode='constant', cval=0.0, crop=None, prefilter=True, axis=None, affine=None, rotate=None, zoom=None)
    image_deformed, mask_deformed = elastic_transform_PIL(image, mask, image_array.shape[1]*2, image_array.shape[1]*0.2, image_array.shape[1]*0.2)
    print("unique:", np.unique(mask_deformed))
    image_elastic_PIL = Image.fromarray((image_deformed).astype('uint8')).convert('L')
    mask_elastic_PIL = Image.fromarray((mask_deformed).astype('uint8')).convert('L')
    #gt_elastic_PIL = Image.fromarray((gt_elastic).astype('uint8')).convert('L')
    image_elastic_PIL.save('./img_elastic.png')
    mask_elastic_PIL.save('./mask_elastic.png')
    #gt_elastic_PIL.save('./gt_elastic.png')
    #
    # cv2.namedWindow("img_a", 0)
    # cv2.imshow("img_a", img_show)
    # cv2.namedWindow("img_c", 0)
    # cv2.imshow("img_c", imageC)
    # cv2.waitKey(0)


# if __name__ == '__main__':
#     img_path = '/home/cxj/Desktop/img/8_5_5.png'
#     imageA = cv2.imread(img_path)
#     img_show = imageA.copy()
#     imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#     # Apply elastic transform on image
#     imageC = elastic_transform(imageA, imageA.shape[1] * 2,
#                                    imageA.shape[1] * 0.08,
#                                    imageA.shape[1] * 0.08)
#
#     cv2.namedWindow("img_a", 0)
#     cv2.imshow("img_a", img_show)
#     cv2.namedWindow("img_c", 0)
#     cv2.imshow("img_c", imageC)
#     cv2.waitKey(0)