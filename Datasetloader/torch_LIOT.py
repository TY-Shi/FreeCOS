import torch
from torch.nn import functional as F
import tifffile
from PIL import Image
import numpy as np
import os
import cv2
import time
import math
import torch.nn as nn
def torch_trans_liot(img):
    '''
    This function is torch liot;
    '''
    #input image N*1*H*W

    #img = np.asarray(img)  # input image H*W*C
    pading = (8,8,8,8,0,0,0,0)
    pad_img = F.pad(img,pading,mode='constant', value=0)#(N,1,h+16,w+16)
    #print("pad_img",pad_img.shape)
    Weight = pad_img.shape[2]
    Height = pad_img.shape[3]
    sum_map = torch.zeros([img.shape[0], 4, img.shape[2],img.shape[3]],dtype=torch.uint8)#N*4*H*W
    direction_map = torch.zeros([img.shape[0], 8, img.shape[2],img.shape[3]],dtype=torch.uint8)#N*8*H*W

    for direction in range(0, 4):
        for postion in range(0, 8):
            if direction == 0:  # Right
                new_pad = pad_img[:,:,postion + 9: Weight - 7 + postion, 8:-8]  # from low to high
            # new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
            elif direction == 1:  # Left
                # new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                new_pad = pad_img[:,:,postion:-1 * (16 - postion), 8:-8]  # from high to low
            elif direction == 2:  # Up
                new_pad = pad_img[:,:,8:-8, postion + 9:Height - 7 + postion]  # from low to high
            # new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
            elif direction == 3:  # Down
                # new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                new_pad = pad_img[:,:,8:-8, postion:-1 * (16 - postion)]  # from high to low
            #print("img_shape",img.shape)
            #print("new_pad_shape",new_pad.shape)
            tmp_map = img.to(torch.int64) - new_pad.to(torch.int64)
            tmp_map[tmp_map > 0] = 1
            tmp_map[tmp_map <= 0] = 0
            #print("tmp_map",tmp_map.shape)
            #print("direction_map",direction_map.shape)
            direction_map[:,postion, :, :] = tmp_map[:,0,:,:] * math.pow(2, postion)
        sum_direction = torch.sum(direction_map, 1)
        sum_map[:,direction, :, :] = sum_direction

    return sum_map

def trans_liot(img):
    '''
    This function is pil liot;
    '''
    #input image H*W

    img = np.asarray(img)  # input image H*W->HWC form
    pad_img = np.pad(img, ((8, 8)), 'constant')#(256+8,256+8)
    #print("pad_img",pad_img.shape)
    Weight = pad_img.shape[0]
    Height = pad_img.shape[1]
    sum_map = np.zeros([4, img.shape[0],img.shape[1]],dtype=np.uint8)#N*4*H*W
    direction_map = np.zeros([8, img.shape[0],img.shape[1]],dtype=np.uint8)#N*8*H*W

    for direction in range(0, 4):
        for postion in range(0, 8):
            if direction == 0:  # Right
                new_pad = pad_img[postion + 9: Weight - 7 + postion, 8:-8]  # from low to high
            # new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
            elif direction == 1:  # Left
                # new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                new_pad = pad_img[postion:-1 * (16 - postion), 8:-8]  # from high to low
            elif direction == 2:  # Up
                new_pad = pad_img[8:-8, postion + 9:Height - 7 + postion]  # from low to high
            # new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
            elif direction == 3:  # Down
                # new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                new_pad = pad_img[8:-8, postion:-1 * (16 - postion)]  # from high to low
            tmp_map = img.astype(np.int64) - new_pad.astype(np.int64)
            tmp_map[tmp_map > 0] = 1
            tmp_map[tmp_map <= 0] = 0
            direction_map[postion, :, :] = tmp_map[:,:] * math.pow(2, postion)
        sum_direction = np.sum(direction_map, 0)
        sum_map[direction, :, :] = sum_direction

    return sum_map
#
def img_region(img):
    #
    img_tensor = torch.from_numpy(img).float()
    img_NXHW = torch.unsqueeze(torch.unsqueeze(img_tensor,0),0)
    #print("img_NXHW",img_NXHW.shape)
    kernel_size = 11
    pad_size = (kernel_size - 1)//2
    m = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=pad_size)
    img_maxpool = m(img_NXHW)
    img_squeeze = torch.squeeze(torch.squeeze(img_maxpool,0),0)
    img_maxpool_array = np.array(img_squeeze)
    return img_maxpool_array


def trans_liot_region(img):
    '''
    This function is pil liot;
    '''
    #input image H*W

    img_array = np.asarray(img)  # input image H*W->HWC form
    img = img_region(img_array)
    pad_img = np.pad(img, ((8, 8)), 'constant')#(256+8,256+8)
    #print("pad_img",pad_img.shape)
    Weight = pad_img.shape[0]
    Height = pad_img.shape[1]
    sum_map = np.zeros([4, img.shape[0],img.shape[1]],dtype=np.uint8)#N*4*H*W
    direction_map = np.zeros([8, img.shape[0],img.shape[1]],dtype=np.uint8)#N*8*H*W

    for direction in range(0, 4):
        for postion in range(0, 8):
            if direction == 0:  # Right
                new_pad = pad_img[postion + 9: Weight - 7 + postion, 8:-8]  # from low to high
            # new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
            elif direction == 1:  # Left
                # new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                new_pad = pad_img[postion:-1 * (16 - postion), 8:-8]  # from high to low
            elif direction == 2:  # Up
                new_pad = pad_img[8:-8, postion + 9:Height - 7 + postion]  # from low to high
            # new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
            elif direction == 3:  # Down
                # new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                new_pad = pad_img[8:-8, postion:-1 * (16 - postion)]  # from high to low
            tmp_map = img.astype(np.int64) - new_pad.astype(np.int64)
            tmp_map[tmp_map > 0] = 1
            tmp_map[tmp_map <= 0] = 0
            direction_map[postion, :, :] = tmp_map[:,:] * math.pow(2, postion)
        sum_direction = np.sum(direction_map, 0)
        sum_map[direction, :, :] = sum_direction

    return sum_map

def trans_liot_region_stride(img,step=6,stride = 5):
    '''
    This function is pil liot;
    '''
    #input image H*W

    img_array = np.asarray(img)  # input image H*W->HWC form
    img = img_region(img_array)
    pad_img = np.pad(img, (((stride+1) * 8, (stride+1) * 8)), 'constant')  # (256+8,256+8)
    Weight = pad_img.shape[0]
    Height = pad_img.shape[1]
    sum_map = np.zeros([4, img.shape[0],img.shape[1]],dtype=np.uint8)#N*4*H*W
    direction_map = np.zeros([8, img.shape[0],img.shape[1]],dtype=np.uint8)#N*8*H*W
    total_length = (stride+1) * 8
    #
    for direction in range(0, 4):
        for postion in range(0, total_length,step):#add step
            #print("postion",postion)
            if direction == 0:  # Right
                new_pad = pad_img[postion + (stride+1) * 9: Weight - (stride+1)*7 + postion, (stride+1) * 8:-8*(stride+1)]  # from low to high
                #print("right_shape",new_pad.shape)
            # new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
            elif direction == 1:  # Left
                # new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                new_pad = pad_img[postion:-1 * (2*8*(stride+1) - postion), (stride+1) * 8:-8*(stride+1)]  # from high to low
                #print("left_shape", new_pad.shape)
            elif direction == 2:  # Up
                new_pad = pad_img[(stride+1) * 8:-8*(stride+1), postion + (stride+1) * 9:Height - (stride+1)*7 + postion]  # from low to high
                #print("up_shape", new_pad.shape)
            # new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
            elif direction == 3:  # Down
                # new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                new_pad = pad_img[(stride+1) * 8:-8*(stride+1), postion:-1 * (2*8*(stride+1) - postion)]  # from high to low
                #print("down_shape", new_pad.shape)
            #print("img_shape",img.shape)
            #print("new_pad_shape",new_pad.shape)
            tmp_map = img.astype(np.int64) - new_pad.astype(np.int64)
            tmp_map[tmp_map > 0] = 1
            tmp_map[tmp_map <= 0] = 0
            #print("tmp_map",tmp_map.shape)
            #print("direction_map",direction_map.shape)
            direction_map[postion//step, :, :] = tmp_map[:,:] * math.pow(2, postion//step)
        sum_direction = np.sum(direction_map, 0)
        #print("sum_direction_shape",sum_direction.shape)
        sum_map[direction, :, :] = sum_direction

    return sum_map

def trans_liot_differentsize(img):
    '''
    This function is pil liot;
    '''
    #input image H*W
    stride_list = [0,2,4]
    img_array = np.asarray(img)  # input image H*W->HWC form
    #img = img_region(img_array)
    #pad_img = np.pad(img, ((8, 8)), 'constant')#(256+8,256+8)
    sum_map = np.zeros([4*len(stride_list), img_array.shape[0], img_array.shape[1]], dtype=np.uint8)  # N*4*H*W
    for stride in stride_list:
        step = stride + 1
        pad_img = np.pad(img_array, (((stride+1) * 8, (stride+1) * 8)), 'constant')  # (256+8,256+8)
        Weight = pad_img.shape[0]
        Height = pad_img.shape[1]
        direction_map = np.zeros([8, img_array.shape[0],img_array.shape[1]],dtype=np.uint8)#N*8*H*W
        total_length = (stride+1) * 8
        #
        for direction in range(0, 4):
            for postion in range(0, total_length,step):#add step
                #print("postion",postion)
                if direction == 0:  # Right
                    new_pad = pad_img[postion + (stride+1) * 9: Weight - (stride+1)*7 + postion, (stride+1) * 8:-8*(stride+1)]  # from low to high
                    #print("right_shape",new_pad.shape)
                # new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
                elif direction == 1:  # Left
                    # new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                    new_pad = pad_img[postion:-1 * (2*8*(stride+1) - postion), (stride+1) * 8:-8*(stride+1)]  # from high to low
                    #print("left_shape", new_pad.shape)
                elif direction == 2:  # Up
                    new_pad = pad_img[(stride+1) * 8:-8*(stride+1), postion + (stride+1) * 9:Height - (stride+1)*7 + postion]  # from low to high
                    #print("up_shape", new_pad.shape)
                # new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
                elif direction == 3:  # Down
                    # new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                    new_pad = pad_img[(stride+1) * 8:-8*(stride+1), postion:-1 * (2*8*(stride+1) - postion)]  # from high to low
                    #print("down_shape", new_pad.shape)
                #print("img_shape",img.shape)
                #print("new_pad_shape",new_pad.shape)
                tmp_map = img_array.astype(np.int64) - new_pad.astype(np.int64)
                tmp_map[tmp_map > 0] = 1
                tmp_map[tmp_map <= 0] = 0
                #print("tmp_map",tmp_map.shape)
                #print("direction_map",direction_map.shape)
                direction_map[postion//step, :, :] = tmp_map[:,:] * math.pow(2, postion//step)
            sum_direction = np.sum(direction_map, 0)
            #print("sum_direction_shape",sum_direction.shape)
            sum_map[direction, :, :] = sum_direction

    return sum_map


if __name__ == "__main__":
    #img = torch.zeros([1, 1, 900,960])
    img = torch.zeros([512, 512])
    time_1 = time.time()
    #liot_img = trans_liot(img)
    liot_img = trans_liot_region_stride(img)
    time_2 = time.time()
    time_delt = time_2 - time_1
    print("time_delt",time_delt)#10-2s 5-1.3s 20-5.73s
    print("liot_img",liot_img.shape)