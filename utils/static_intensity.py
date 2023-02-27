import turtle
import numpy as np
import random
import math
from PIL import Image
import os
from skimage import morphology
#Img_dir = "./DRIVE_test/1st_manual/"
#Orgin_dir = "./DRIVE_test/images/"
Img_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/test/img/"
gt_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/test/gt/"

# Img_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_grayvessel_width/"
# gt_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_gtvessel_width/"

# Orgin_dir = "./DRHAGIS/all_resize/images/"
# #Img_dir = "./"+ "CrackTree_image_resize_640_800/"
# Save_dir  ="./"+ "DRIVE_vessel/"
# Save_RGB_dir = "./DRIVE_RGB_vessel/"
# Save_Distance_dir = "./DRIVE_Distance_vessel/"
# if not os.path.exists(Save_dir):
#     os.makedirs(Save_dir)
# if not os.path.exists(Save_RGB_dir):
#     os.makedirs(Save_RGB_dir)
# if not os.path.exists(Save_Distance_dir):
#     os.makedirs(Save_Distance_dir)
files = os.listdir(Img_dir)
#caluate diameter
total_images = np.zeros((512,512,20))
i = 0
sum_list = []
sum_array = np.zeros((1,1))
sum_array = np.reshape(sum_array, -1)
dict_intensity = {}
for file in files:
    file_name = os.path.join(Img_dir,file)
    gt_name  = os.path.join(gt_dir,file)
    image = Image.open(file_name).convert("L")
    gt = Image.open(gt_name).convert("L")

    image_array = np.asarray(image).astype(np.uint64)
    gt_array = np.asarray(gt)
    print("unique_array_before", np.unique(image_array))
    print("unique_gt",np.unique(gt_array))
    image_array[gt_array==0] = 1000
    print("unique_array",np.unique(image_array))
    values, counts = np.unique(image_array, return_counts=True)
    values_key = values.tolist()
    counts_list = counts.tolist()
    for idx, intensity in enumerate(values_key):
        #print("intensity",intensity)
        if intensity not in dict_intensity.keys():
            dict_intensity[intensity] = counts_list[idx]
        else:
            dict_intensity[intensity] += counts_list[idx]
print("len(files)",len(files))
for key in dict_intensity.keys():
    print("key:{},intensity:{}",key,dict_intensity[key])

del dict_intensity[1000]
print("del_1000^^^^^^^^^^^")
# for key in dict_intensity.keys():
#     print("key:{},intensity:{}",key,dict_intensity[key])
# 60

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
values = [] #in same order as traversing keys
keys = [] #also needed to preserve order
for key in dict_intensity.keys():
  keys.append(key)
  values.append(dict_intensity[key])
plt.bar(keys, values, color='g')
#plt.hist(numpy_list, bins = [1.,2.,3.,4.,5.,6.,7.,8.])
plt.savefig("hist_fake_vessel.png",dpi=400,bbox_inches='tight')
#     orgin_convert = np.ones((image_array.shape[0],image_array.shape[1],3))
#     print("image_array",image_array.shape)
#     print("image_array", np.max(image_array))
#     print("image_array", np.min(image_array))
#     image_convert = np.ones_like(image_array) * 255
#     image_convert[image_array==255] = 0
#     orgin_convert[:,:,0] = orgin_array[:, :, 0]
#     orgin_convert[:, :, 1] = orgin_array[:, :, 1]
#     orgin_convert[:, :, 2] = orgin_array[:, :, 2]
#     orgin_convert[:,:,0][image_convert==255] = 255
#     orgin_convert[:, :, 1][image_convert == 255] = 255
#     orgin_convert[:, :, 2][image_convert == 255] = 255
#     print("unique",np.unique(image_convert))
#     save_distance_path = Save_Distance_dir + save_name
#     skel, distance = morphology.medial_axis(image_array, return_distance=True)
#     print("distance_map",distance.shape)
#     print("unique_distance",np.unique(distance))
#     print("unique_skel", np.unique(skel))
#     #Image.fromarray((distance*255).astype('uint8')).convert('L').save(save_distance_path)
#     dist_on_skel = distance * skel
#     print("dist_on_skel", np.unique(dist_on_skel))
#     dist_int = np.rint(dist_on_skel)
#     reshape_distancemap = np.reshape(dist_int, -1)
#     sum_list.append(reshape_distancemap)
#     print("sum_array",sum_array.shape)
#     print("reshape",reshape_distancemap.shape)
#
#     sum_array = np.concatenate((sum_array,reshape_distancemap))
#     #total_images[:,:,i] = dist_int
#     i += 1
#     #hist, bins = np.histogram(dist_int, bins=np.unique(dist_int))
#     skel_img = (((dist_on_skel-np.min(dist_on_skel))/(np.max(dist_on_skel)-np.min(dist_on_skel)))*255).astype(np.uint8)
#     #Image.fromarray((skel_img).astype('uint8')).convert('L').save(save_distance_path)
#
# from matplotlib import pyplot as plt
# print("list",np.unique(sum_array))
# #numpy_list = np.reshape(total_images,-1)
# int_sum_array = np.array(list(map(np.int, np.unique(sum_array))))
# print("int_sum_array",int_sum_array)
# numpy_list = sum_array
# plt.hist(numpy_list, bins = int_sum_array.tolist()[1:])
# #plt.hist(numpy_list, bins = [1.,2.,3.,4.,5.,6.,7.,8.])
# plt.title("histogram")
# plt.show()
# hist, bins = np.histogram(numpy_list, bins=[1.,2.,3.,4.,5.,6.,7.,8.])
# print("hist",hist)