import numpy as np
import random
import math
from PIL import Image
import os


def replace_color_binary(img, src_clr = [255,255,255],dst_clr = [0,0,0]):
    img_arr = np.asarray(img, dtype=np.double)

    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]

    r_img[img > src_color] = dst_clr[0]
    g_img[img > src_color] = dst_clr[1]
    b_img[img > src_color] = dst_clr[2]

    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)

    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img

#Img_dir = "./DRIVE_test/1st_manual/"
#Orgin_dir = "./DRIVE_test/images/"
# Img_dir = "./train/fake_rgbvessel_more/"
# Save_dir  ="./train/fake_vessel_gt_more/"
Img_dir = "./train/fake_onlythin_vessel/"
Save_dir  ="./train/fake_onlythin_gt/"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
files = os.listdir(Img_dir)
i = 0
list_RGB= []
idx = 300
for image_dir in files:
    print("image_dir",image_dir)
    image =  Image.open(Img_dir+image_dir).convert("L")
    image_array = np.asarray(image).copy()
    print("image_array", image_array.shape)
    #print("unique",np.unique(image_array))

    image_array[image_array==255] = 0
    image_array[image_array>0] = 255
    print("unique",np.unique(image_array))
    #image_array[image_array==255] = 0
    #binary = replace_color_binary(image_array)
    save_path = Save_dir + str(idx) + '.png'
    idx += 1
    Image.fromarray((image_array).astype('uint8')).convert('L').save(save_path)