import numpy as np
from PIL import Image, ImageEnhance
#from utils import FDA_source_to_target_np
from __init__ import FDA_source_to_target_np
import os
import scipy.misc
from PIL import ImageFilter
import math
import cv2
import random
def _rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """以给定的角，旋转一张图片，设置输出大小。根据需要旋转是否去除黑边。
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.
    Returns:
        A rotated image.
    """

    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:

        #image = image.rotate(math.radians(rotation_degree), Image.NEAREST, expand=False)
        image = image.rotate(rotation_degree, Image.NEAREST, expand=False)

        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = _largest_rotated_rect(output_height, output_width, math.radians(rotation_degree))
            box = (output_width / 2.0 - lrr_width / 2.0, (output_height - lrr_height) / 2,
                   output_width / 2.0 + lrr_width / 2.0, output_height / 2.0 + lrr_height / 2.0)
            resized_image = image.crop(box)
            image = resized_image.resize((output_height, output_width))

    return image


def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    #row, col, _ = X_imgs[0].shape
    row, col, _ = X_imgs.shape
    # Gaussian distribution parameters
    mean = 0
    var = 3
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))  # np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(X_imgs.shape, np.float32)

    if len(X_imgs.shape) == 2:
        noisy_image = X_imgs + gaussian
    else:
        noisy_image[:, :, 0] = X_imgs[:, :, 0] + gaussian
        noisy_image[:, :, 1] = X_imgs[:, :, 1] + gaussian
        noisy_image[:, :, 2] = X_imgs[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def _largest_rotated_rect(w, h, angle):
    # 计算中心最大矩形区域
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def replace_color(img, src_clr = [255,255,255],dst_clr = [0,0,0]):
    img_arr = np.asarray(img, dtype=np.double)

    # 分离通道
    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    # 编码
    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]

    # 索引并替换颜色
    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]

    # 合并通道
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    # 将数据转换为图像数据(h,w,c)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img

def replace_color_binary(img, src_clr = [0,0,0],dst_clr = [255,255,255]):
    img_arr = np.asarray(img, dtype=np.double)

    # 分离通道
    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    # 编码
    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]

    # 索引并替换颜色
    r_img[img > src_color] = dst_clr[0]
    g_img[img > src_color] = dst_clr[1]
    b_img[img > src_color] = dst_clr[2]

    # 合并通道
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    # 将数据转换为图像数据(h,w,c)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img

src_file = "D://Project_code//Generator_vessel//make_fake_vessel_code//FDA//FDA_retinalIOSTAR_image//"
label_file = "D://Project_code//Generator_vessel//make_fake_vessel_code//FDA//FDA_retinalIOSTAR_label//"
Save_image = "./FDA_IOSTAR_filter_image/"
Save_label = "./FDA_IOSTAR_filter_label/"
if not os.path.exists(Save_image):
    os.makedirs(Save_image)
if not os.path.exists(Save_label):
    os.makedirs(Save_label)
# im_src = Image.open("src_in_tar_retina_label.png").convert('L')
# im_trg = Image.open("21_training.ppm").convert('RGB')
# print("image_src",np.asarray(im_src).shape)
# print(np.unique(np.asarray(im_src)))
files_src = os.listdir(src_file)
# c = os.path.join("./a","b.png")
# print("c",c)
counter = 0
for i in range(len(files_src)):
    vessel_image_path = src_file + files_src[i]
    label_image_path = label_file + files_src[i]
    img = Image.open(vessel_image_path).convert('RGB')
    label = Image.open(label_image_path).convert('RGB')
    save_image_path = Save_image + str(counter) + ".png"
    save_label_path = Save_label + str(counter) + ".png"
    img = np.asarray(img)
    label = np.asarray(label)
    Image.fromarray((img).astype('uint8')).convert('L').save(save_image_path)
    Image.fromarray((label).astype('uint8')).convert('L').save(save_label_path)
    counter += 1
#scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('src_in_tar.png')

