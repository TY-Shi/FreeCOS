import numpy as np
from PIL import Image
#from utils import FDA_source_to_target_np
from __init__ import FDA_source_to_target_np
import scipy.misc


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
#im_src = Image.open("demo_images/source.png").convert('RGB')
#im_trg = Image.open("demo_images/target.png").convert('RGB')
im_src = Image.open("work_vessel.png").convert('RGB')
im_trg = Image.open("21_training.ppm").convert('RGB')


im_trg = im_trg.crop((81,142,479,502))
im_src = im_src.resize( (512,512), Image.BICUBIC )
im_trg = im_trg.resize( (512,512), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))
print("im_src",np.max(im_src))
print("im_trg",np.max(im_trg))
src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.4 )
im_label = im_src.copy()


src_in_trg = src_in_trg.transpose((1,2,0))
im_label = im_label.transpose((1,2,0))
im_label = replace_color(im_label)
im_label = replace_color_binary(im_label)
# im_label[:,:,0][im_label[:,:,0]>0] = 255
# im_label[:,:,1][im_label[:,:,1]>0] = 255
# im_label[:,:,2][im_label[:,:,2]>0] = 255
print("im+label",im_label.shape)
print("im+src_in_trg",src_in_trg.shape)
print("src_in_trg",np.max(src_in_trg))
print("im_label_max",np.max(im_label))
img_FDA = np.clip(src_in_trg,0,255.)
im_label = np.clip(im_label,0,255.)
Image.fromarray((img_FDA).astype('uint8')).convert('RGB').save('src_in_tar_vessel.png')
Image.fromarray((im_label).astype('uint8')).convert('L').save('src_in_tar_vessel_label.png')
#scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('src_in_tar.png')

