import numpy as np
from PIL import Image
#from utils import FDA_source_to_target_np
#from __init__ import FDA_source_to_target_np
import scipy.misc

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


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

    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

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
im_src = Image.open("reverse_0.png").convert('RGB')
im_trg = Image.open("./img/000_PPA_44_PSA_00_8.png").convert('RGB')


#im_trg = im_trg.crop((81,142,479,502))
im_src = im_src.resize( (512,512), Image.BICUBIC )
im_trg = im_trg.resize( (512,512), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)
print("im_src",im_src.shape)
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
#im_label = np.clip(im_label,0,255.)
Image.fromarray((img_FDA).astype('uint8')).convert('L').save('src_in_tar_vessel.png')
#Image.fromarray((im_label).astype('uint8')).convert('L').save('src_in_tar_vessel_label.png')
#scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('src_in_tar.png')

