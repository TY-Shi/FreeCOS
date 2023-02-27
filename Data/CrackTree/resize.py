import numpy as np
import random
import math
from PIL import Image
import os
#Img_dir = "./DRIVE_test/1st_manual/"
#Orgin_dir = "./DRIVE_test/images/"
#Img_dir = "./fake_rgbvessel/"
Img_dir = "./test/img/"
Gt_dir = "./test/gt/"
Save_dir  ="./test/img_resize/"
Savegt_dir  ="./test/gt_resize/"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
if not os.path.exists(Savegt_dir):
    os.makedirs(Savegt_dir)

files = os.listdir(Img_dir)
gt_file = os.listdir(Gt_dir)
i = 0
list_RGB= []
for image_dir in files:
    print("image_dir",image_dir)
    image =  Image.open(Img_dir+image_dir).convert("L")
    out = image.resize((512, 512))
    image_array = np.asarray(out)
    print("image_array",image_array.shape)
    save_path = Save_dir + image_dir
    Image.fromarray((image_array).astype('uint8')).convert('L').save(save_path)

for image_dir in gt_file:
    print("image_dir",image_dir)
    image =  Image.open(Gt_dir+image_dir).convert("L")
    out = image.resize((512, 512))
    image_array = np.asarray(out)
    print("image_array",image_array.shape)
    save_path = Savegt_dir + image_dir
    Image.fromarray((image_array).astype('uint8')).convert('L').save(save_path)

