import numpy as np
import cv2
import PIL.Image as Image

def replace_color(img, src_clr = [255,255,255],dst_clr = [0,0,0]):
    img_arr = np.asarray(img, dtype=np.double)

    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]

    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]

    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img

filepath = "./fake_vessel/00000.png"
img = np.array(Image.open(filepath).convert('L'))
img[img > 0] = 1
print("img_unique",np.unique(img))
img_reverse = img*(-1)+1
img_reverse = img_reverse *255
im=Image.fromarray(img_reverse).convert('L')
im.save("reverse_0.png")
print("img_shape",img.shape)
print("img_reverse",np.unique(img_reverse))