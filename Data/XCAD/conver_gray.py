import numpy as np
import random
import math
from PIL import Image
import os
#Img_dir = "./DRIVE_test/1st_manual/"
#Orgin_dir = "./DRIVE_test/images/"
#Img_dir = "./fake_rgbvessel/"
#Img_dir = "./fake_rgbvessel_width/"
#Img_dir = "./fake_lrange_vessel/"
Img_dir = "./fake_very_smalltheta/"
Save_dir  ="./train/fake_grayvessel_verysmalltheta//"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
files = os.listdir(Img_dir)
i = 0
list_RGB= []
for image_dir in files:
    print("image_dir",image_dir)
    image =  Image.open(Img_dir+image_dir).convert("L")
    image_array = np.asarray(image)
    print("image_array",image_array.shape)
    save_path = Save_dir + image_dir
    Image.fromarray((image_array).astype('uint8')).convert('L').save(save_path)