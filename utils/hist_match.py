import numpy as np
import random
import math
from PIL import Image
import os
from skimage import morphology
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import torch
def histMatch(grayArray,h_d):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des

def src_trg_histMatch(h_src,h_trg):
    #计算累计直方图
    tmp = 0.0
    h_acc_trg = h_trg.copy()
    for i in range(255):
        if i in h_trg.keys():
            tmp += h_trg[i]
            h_acc_trg[i] = tmp

    tmp = 0.0
    h_acc_src = h_src.copy()
    #except 255
    for i in range(255):
        if i in h_acc_src.keys():
            tmp += h_src[i]
            h_acc_src[i] = tmp
    h_acc_src_max_key = max(h_acc_src, key=h_acc_src.get)
    #print("h_acc_src_key", h_acc_src)
    for key in h_acc_src.keys():
        h_acc_src[key] = h_acc_src[key]/h_acc_src[h_acc_src_max_key]
    h_acc_trg_max_key = max(h_acc_trg, key=h_acc_trg.get)
    #print("h_acc_trg_max_key",h_acc_trg_max_key)
    for key in h_acc_trg.keys():
        h_acc_trg[key] = h_acc_trg[key]/h_acc_trg[h_acc_trg_max_key]
    #计算映射
    M = np.zeros(256)
    for i in range(255):
        idx = 0
        minv = 1
        #find the nears
        if i in h_acc_src.keys():
            for j in h_acc_trg.keys():
                #print(h_acc_trg[j] - h_acc_src[i])
                if (j in h_acc_trg.keys()) and (np.fabs(h_acc_trg[j] - h_acc_src[i]) < minv):
                    minv = np.fabs(h_acc_trg[j] - h_acc_src[i])
                    idx = int(j)
            M[i] = idx
    M[255] = 255
    # print("h_acc_trg",h_acc_trg)
    # print("h_acc_src", h_acc_src)
    #print("M_matrix",M)
    #des = M[grayArray]
    return M

def src_trg_histMatch_whole(h_src,h_trg):
    #计算累计直方图
    tmp = 0.0
    h_acc_trg = h_trg.copy()
    for i in range(256):
        if i in h_trg.keys():
            tmp += h_trg[i]
            h_acc_trg[i] = tmp

    tmp = 0.0
    h_acc_src = h_src.copy()
    #except 255
    for i in range(256):
        if i in h_acc_src.keys():
            tmp += h_src[i]
            h_acc_src[i] = tmp
    h_acc_src_max_key = max(h_acc_src, key=h_acc_src.get)
    #print("h_acc_src_key", h_acc_src)
    for key in h_acc_src.keys():
        h_acc_src[key] = h_acc_src[key]/h_acc_src[h_acc_src_max_key]
    h_acc_trg_max_key = max(h_acc_trg, key=h_acc_trg.get)
    #print("h_acc_trg_max_key",h_acc_trg_max_key)
    for key in h_acc_trg.keys():
        h_acc_trg[key] = h_acc_trg[key]/h_acc_trg[h_acc_trg_max_key]
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        #find the nears
        if i in h_acc_src.keys():
            for j in h_acc_trg.keys():
                #print(h_acc_trg[j] - h_acc_src[i])
                if (j in h_acc_trg.keys()) and (np.fabs(h_acc_trg[j] - h_acc_src[i]) < minv):
                    minv = np.fabs(h_acc_trg[j] - h_acc_src[i])
                    idx = int(j)
            M[i] = idx
    #M[255] = 255
    # print("h_acc_trg",h_acc_trg)
    # print("h_acc_src", h_acc_src)
    #print("M_matrix",M)
    #des = M[grayArray]
    return M


def get_target_dataset_hist(Img_dir,gt_dir):
    dict_intensity = {}
    files = os.listdir(Img_dir)
    for file in files:
        file_name = os.path.join(Img_dir, file)
        gt_name = os.path.join(gt_dir, file)
        image = Image.open(file_name).convert("L")
        gt = Image.open(gt_name).convert("L")

        image_array = np.asarray(image).astype(np.uint64)
        gt_array = np.asarray(gt)
        #print("unique_array_before", np.unique(image_array))
        #print("unique_gt", np.unique(gt_array))
        image_array[gt_array == 0] = 1000
        #print("unique_array", np.unique(image_array))
        values, counts = np.unique(image_array, return_counts=True)
        values_key = values.tolist()
        counts_list = counts.tolist()
        for idx, intensity in enumerate(values_key):
            # print("intensity",intensity)
            if intensity not in dict_intensity.keys():
                dict_intensity[intensity] = counts_list[idx]
            else:
                dict_intensity[intensity] += counts_list[idx]
    #print("len(files)", len(files))
    # for key in dict_intensity.keys():
    #     print("key:{},intensity:{}", key, dict_intensity[key])

    del dict_intensity[1000]
    #print("del_1000^^^^^^^^^^^")
    return dict_intensity

def get_target_dataset_hist_whole(Img_dir,gt_dir):
    dict_intensity = {}
    files = os.listdir(Img_dir)
    for file in files:
        file_name = os.path.join(Img_dir, file)
        gt_name = os.path.join(gt_dir, file)
        image = Image.open(file_name).convert("L")
        gt = Image.open(gt_name).convert("L")

        image_array = np.asarray(image).astype(np.uint64)
        gt_array = np.asarray(gt)
        #print("unique_array_before", np.unique(image_array))
        #print("unique_gt", np.unique(gt_array))
        #print("unique_array", np.unique(image_array))
        values, counts = np.unique(image_array, return_counts=True)
        values_key = values.tolist()
        counts_list = counts.tolist()
        for idx, intensity in enumerate(values_key):
            # print("intensity",intensity)
            if intensity not in dict_intensity.keys():
                dict_intensity[intensity] = counts_list[idx]
            else:
                dict_intensity[intensity] += counts_list[idx]
    #print("len(files)", len(files))
    # for key in dict_intensity.keys():
    #     print("key:{},intensity:{}", key, dict_intensity[key])
    #print("del_1000^^^^^^^^^^^")
    return dict_intensity

def get_target_dataset_hist_wholeimg(Img_dir):
    dict_intensity = {}
    files = os.listdir(Img_dir)
    for file in files:
        file_name = os.path.join(Img_dir, file)
        image = Image.open(file_name).convert("L")
        image_array = np.asarray(image).astype(np.uint64)
        values, counts = np.unique(image_array, return_counts=True)
        values_key = values.tolist()
        counts_list = counts.tolist()
        for idx, intensity in enumerate(values_key):
            # print("intensity",intensity)
            if intensity not in dict_intensity.keys():
                dict_intensity[intensity] = counts_list[idx]
            else:
                dict_intensity[intensity] += counts_list[idx]
    return dict_intensity

def get_target_dataset_hist_changesrc(Img_dir,gt_dir,M_list):
    dict_intensity = {}
    files = os.listdir(Img_dir)
    for file in files:
        file_name = os.path.join(Img_dir, file)
        gt_name = os.path.join(gt_dir, file)
        image = Image.open(file_name).convert("L")
        gt = Image.open(gt_name).convert("L")

        image_array = np.asarray(image).astype(np.uint64)
        gt_array = np.asarray(gt)
        for key in range(len(M_list)):
            image_array[image_array == key] = M_list[key]
        #print("unique_array_before", np.unique(image_array))
        #print("unique_gt", np.unique(gt_array))
        image_array[gt_array == 0] = 1000
        #print("unique_array", np.unique(image_array))
        values, counts = np.unique(image_array, return_counts=True)
        values_key = values.tolist()
        counts_list = counts.tolist()
        for idx, intensity in enumerate(values_key):
            # print("intensity",intensity)
            if intensity not in dict_intensity.keys():
                dict_intensity[intensity] = counts_list[idx]
            else:
                dict_intensity[intensity] += counts_list[idx]
    #print("len(files)", len(files))
    # for key in dict_intensity.keys():
    #     print("key:{},intensity:{}", key, dict_intensity[key])

    del dict_intensity[1000]
    #print("del_1000^^^^^^^^^^^")
    return dict_intensity

def get_target_dataset_hist_tensor(Img_tensor,Pred_tensor,dict_intensity_epoch):
    #Img_tensor:NCHW
    #gt_tensor:NCHW
    #dict_intensity_epoch = {}
    Pred_tensor = Pred_tensor.detach().cpu()
    pred_mask = torch.where(Pred_tensor >= 0.5, 1, 0)
    pred_array = np.asarray(pred_mask).astype(np.uint64)
    image_array = np.asarray(Img_tensor.detach().cpu())
    #print("unique_array_before", np.unique(image_array))
    #print("unique_gt", np.unique(gt_array))
    image_array = (image_array * 255).astype(np.uint64)
    #print("unique_array", np.unique(image_array))
    image_array[pred_array==0] = 1000 #background:1000 only caluate vessel region
    values, counts = np.unique(image_array, return_counts=True)
    values_key = values.tolist()
    counts_list = counts.tolist()
    for idx, intensity in enumerate(values_key):
        # print("intensity",intensity)
        if intensity not in dict_intensity_epoch.keys():
            dict_intensity_epoch[intensity] = counts_list[idx]
        else:
            dict_intensity_epoch[intensity] += counts_list[idx]
    del dict_intensity_epoch[1000]
    #print("del_1000^^^^^^^^^^^")
    return dict_intensity_epoch

def get_target_dataset_hist_tensor_whole(Img_tensor,dict_intensity_epoch):
    #Img_tensor:NCHW
    #gt_tensor:NCHW
    #dict_intensity_epoch = {}
    image_array = np.asarray(Img_tensor.detach().cpu())
    #print("unique_array_before", np.unique(image_array))
    #print("unique_gt", np.unique(gt_array))
    image_array = (image_array * 255).astype(np.uint64)
    values, counts = np.unique(image_array, return_counts=True)
    values_key = values.tolist()
    counts_list = counts.tolist()
    for idx, intensity in enumerate(values_key):
        # print("intensity",intensity)
        if intensity not in dict_intensity_epoch.keys():
            dict_intensity_epoch[intensity] = counts_list[idx]
        else:
            dict_intensity_epoch[intensity] += counts_list[idx]
    return dict_intensity_epoch

def drawHist(hist,name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist)-1#x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    #plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys),tuple(values))#绘制直方图

def arrayToHist(grayArray,nums):
    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #normalize
    #n = w*h
    n = 1
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist

def comput_static_transMatrix(trg_Img_dir,trg_gt_dir,src_Img_dir,src_gt_dir):
    trg_hist = get_target_dataset_hist(trg_Img_dir, trg_gt_dir)
    src_hist = get_target_dataset_hist(src_Img_dir, src_gt_dir)
    trans_M = src_trg_histMatch(src_hist, trg_hist)  # 0-255
    trans_M_list = trans_M.tolist()
    return trans_M_list

def comput_static_transMatrix_whole(trg_Img_dir,trg_gt_dir,src_Img_dir,src_gt_dir):
    trg_hist = get_target_dataset_hist_whole(trg_Img_dir, trg_gt_dir)
    src_hist = get_target_dataset_hist_whole(src_Img_dir, src_gt_dir)
    trans_M = src_trg_histMatch_whole(src_hist, trg_hist)  # 0-255
    trans_M_list = trans_M.tolist()
    return trans_M_list

if __name__ == '__main__':
    imdir = "./hw1_s2.jpg"
    imdir_match = "./hw1_s22.jpg"
    src_Img_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_grayvessel_width/"
    src_gt_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_gtvessel_width/"
    trg_Img_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/test/img/"
    trg_gt_dir = "/mnt/nas/sty/codes/Unsupervised/Data/XCAD/test/gt/"
    trg_hist = get_target_dataset_hist(trg_Img_dir, trg_gt_dir)
    src_hist = get_target_dataset_hist(src_Img_dir, src_gt_dir)
    trans_M = src_trg_histMatch(src_hist, trg_hist)#0-255



    #直方图匹配
    #打开文件并灰度化
    source_img = src_Img_dir + '103.png'
    im_s = Image.open(source_img).convert("L")
    im_s = np.array(im_s)
    print("len_N",trans_M.shape)
    trans_M_list = trans_M.tolist()

    src_change_hist = get_target_dataset_hist_changesrc(src_Img_dir, src_gt_dir, trans_M_list)
    keys = []
    values = []
    for key in src_change_hist.keys():
        keys.append(key)
        values.append(src_change_hist[key])
    plt.bar(keys, values, color='g')
    # plt.hist(numpy_list, bins = [1.,2.,3.,4.,5.,6.,7.,8.])
    plt.savefig("hist_srcchange_vessel.png", dpi=400, bbox_inches='tight')

    # des_src = im_s.copy()
    # for key in range(len(trans_M_list)):
    #     des_src[des_src==key] = trans_M_list[key]
    # #print("des_src",np.unique(des_src))
    # #print("im_s",np.unique(im_s))
    #
    # #trans_M(im_s)
    # #print("des_source",des_source.shape)
    # # print(np.shape(im_s))
    # # #打开文件并灰度化
    # # im_match = Image.open(imdir_match).convert("L")
    # # im_match = np.array(im_match)
    # # print(np.shape(im_match))
    # # #开始绘图
    # # plt.figure()
    # #
    # # #原始图和直方图
    # # plt.subplot(2,3,1)
    # # plt.title("原始图片")
    # # plt.imshow(im_s,cmap='gray')
    # #
    # # plt.subplot(2,2,3)
    # hist_s = arrayToHist(im_s,255)
    # # print("hist_s",hist_s)
    # #drawHist(hist_s,"原始直方图")
    # values = []  # in same order as traversing keys
    # keys = []  # also needed to preserve order
    # del hist_s[255]
    # for key in hist_s.keys():
    #     keys.append(key)
    #     values.append(hist_s[key])
    # plt.bar(keys, values, color='g')
    # # plt.hist(numpy_list, bins = [1.,2.,3.,4.,5.,6.,7.,8.])
    # plt.savefig("hist_original_vessel.png", dpi=400, bbox_inches='tight')
    #
    # #
    # # #match图和其直方图
    # # plt.subplot(2,3,2)
    # # plt.title("match图片")
    # # plt.imshow(im_match,cmap='gray')
    # #
    # # plt.subplot(2,3,5)
    # # hist_m = arrayToHist(im_match,256)
    # # drawHist(hist_m,"match直方图")
    # #
    # # #match后的图片及其直方图
    # #im_d = histMatch(im_s,hist_m)#将目标图的直方图用于给原图做均衡，也就实现了match
    # # plt.subplot(2,2,2)
    # # plt.title("match后的图片")
    # # plt.imshow(des_src,cmap='gray')
    #
    # plt.subplot(2,2,4)
    # hist_d = arrayToHist(des_src,255)
    # # print("hist_d",hist_d)
    # # drawHist(hist_d,"match后的直方图")
    # # plt.savefig("hist_change.png", dpi=400, bbox_inches='tight')
    #
    # values = []  # in same order as traversing keys
    # keys = []  # also needed to preserve order
    # del hist_d[255]
    # for key in hist_d.keys():
    #     keys.append(key)
    #     values.append(hist_d[key])
    # plt.bar(keys, values, color='g')
    # # plt.hist(numpy_list, bins = [1.,2.,3.,4.,5.,6.,7.,8.])
    # plt.savefig("hist_change_vessel.png", dpi=400, bbox_inches='tight')
    # #
    # # plt.show()