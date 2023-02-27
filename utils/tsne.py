"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import torch
import random

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1],str("o"), color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    # mid = int(data.shape[0] / 2)
    # for i in range(mid):
    #     plt.text(data[i, 0], data[i, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
    #              color=plt.cm.Set1(1 / 10),  # 表示颜色
    #              fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
    # for i in range(mid):
    #     plt.text(data[mid + i, 0], data[mid + i, 1], str("o"),
    #              color=plt.cm.Set1(2 / 10),
    #              fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    #xticks(locs, [labels], **kwargs)#locs，是一个数组或者列表，表示范围和刻度，label表示每个刻度的标签
    plt.yticks([0, 1.1])
    #plt.title()
    plt.savefig("./test_tsne.png")
    #plt.show()

def plot_embedding_labelunlabel(label_feature, unlabel_feature, label, unlabel):
    x_min, x_max = np.min(label_feature, 0), np.max(label_feature, 0)
    label_feature = (label_feature - x_min) / (x_max - x_min)
    x_min, x_max = np.min(unlabel_feature, 0), np.max(unlabel_feature, 0)
    unlabel_feature = (unlabel_feature - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(label_feature.shape[0]):
        plt.text(label_feature[i, 0], label_feature[i, 1],str("o"), color="blue",
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(unlabel_feature.shape[0]):
        plt.text(unlabel_feature[i, 0], unlabel_feature[i, 1],str("."), color="red",
                 fontdict={'weight': 'bold', 'size': 9})
    # mid = int(data.shape[0] / 2)
    # for i in range(mid):
    #     plt.text(data[i, 0], data[i, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
    #              color=plt.cm.Set1(1 / 10),  # 表示颜色
    #              fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
    # for i in range(mid):
    #     plt.text(data[mid + i, 0], data[mid + i, 1], str("o"),
    #              color=plt.cm.Set1(2 / 10),
    #              fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    plt.yticks([0, 1.1])
    #plt.title()
    plt.savefig("./test_tsne.png")
    #plt.show()

def plot_embedding_labelunlabel_posneg(label_feature_pos, unlabel_feature_pos, label_feature_neg, unlabel_feature_neg,name):
    x_min, x_max = np.min(label_feature_pos, 0), np.max(label_feature_pos, 0)
    label_feature_pos = (label_feature_pos - x_min) / (x_max - x_min)
    x_min, x_max = np.min(unlabel_feature_pos, 0), np.max(unlabel_feature_pos, 0)
    unlabel_feature_pos = (unlabel_feature_pos - x_min) / (x_max - x_min)

    x_min, x_max = np.min(label_feature_neg, 0), np.max(label_feature_neg, 0)
    label_feature_neg = (label_feature_neg - x_min) / (x_max - x_min)
    x_min, x_max = np.min(unlabel_feature_neg, 0), np.max(unlabel_feature_neg, 0)
    unlabel_feature_neg = (unlabel_feature_neg - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(label_feature_pos.shape[0]):
        plt.text(label_feature_pos[i, 0], label_feature_pos[i, 1],str("o"), color="blue",
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(unlabel_feature_pos.shape[0]):
        plt.text(unlabel_feature_pos[i, 0], unlabel_feature_pos[i, 1],str("o"), color="red",
                 fontdict={'weight': 'bold', 'size': 9})

    for i in range(label_feature_neg.shape[0]):
        plt.text(label_feature_neg[i, 0], label_feature_neg[i, 1],str("o"), color="green",
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(unlabel_feature_neg.shape[0]):
        plt.text(unlabel_feature_neg[i, 0], unlabel_feature_neg[i, 1],str("o"), color="yellow",
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    plt.yticks([0, 1.1])
    #plt.title()
    save_path = "./"+name+".png"
    plt.savefig(save_path)
    #plt.show()

def plot_embedding_labelunlabel_two(label_feature_pos, unlabel_feature_pos, name):
    x_min, x_max = np.min(label_feature_pos, 0), np.max(label_feature_pos, 0)
    label_feature_pos = (label_feature_pos - x_min) / (x_max - x_min)
    x_min, x_max = np.min(unlabel_feature_pos, 0), np.max(unlabel_feature_pos, 0)
    unlabel_feature_pos = (unlabel_feature_pos - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(label_feature_pos.shape[0]):
        plt.text(label_feature_pos[i, 0], label_feature_pos[i, 1],str("o"), color="blue",
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(unlabel_feature_pos.shape[0]):
        plt.text(unlabel_feature_pos[i, 0], unlabel_feature_pos[i, 1],str("o"), color="red",
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    plt.yticks([0, 1.1])
    #plt.title()
    save_path = "./"+name+".png"
    plt.savefig(save_path)



def feature_tsne(label_feature, unlabel_feature, label, unlabel):
    #data, label, n_samples, n_features = get_data()

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    label_feature_result = tsne.fit_transform(label_feature)
    unlabel_feature_result = tsne.fit_transform(unlabel_feature)
    print("label_feature_result",label_feature_result.shape)
    print("unlabel_feature_result",unlabel_feature_result.shape)
    plot_embedding_labelunlabel(label_feature_result, unlabel_feature_result, label, unlabel)


def plot_embedding_labelunlabel(label_feature, unlabel_feature, label, unlabel):
    x_min, x_max = np.min(label_feature, 0), np.max(label_feature, 0)
    label_feature = (label_feature - x_min) / (x_max - x_min)
    x_min, x_max = np.min(unlabel_feature, 0), np.max(unlabel_feature, 0)
    unlabel_feature = (unlabel_feature - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(label_feature.shape[0]):
        plt.text(label_feature[i, 0], label_feature[i, 1],str("o"), color="blue",
                 fontdict={'weight': 'bold', 'size': 9})
    for i in range(unlabel_feature.shape[0]):
        plt.text(unlabel_feature[i, 0], unlabel_feature[i, 1],str("."), color="red",
                 fontdict={'weight': 'bold', 'size': 9})
    # mid = int(data.shape[0] / 2)
    # for i in range(mid):
    #     plt.text(data[i, 0], data[i, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
    #              color=plt.cm.Set1(1 / 10),  # 表示颜色
    #              fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
    # for i in range(mid):
    #     plt.text(data[mid + i, 0], data[mid + i, 1], str("o"),
    #              color=plt.cm.Set1(2 / 10),
    #              fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([-0.1, 1.1])  # 坐标轴设置
    plt.yticks([0, 1.1])
    #plt.title()
    plt.savefig("./test_tsne.png")

def feature_tsne_posneg(label_feature_pos, unlabel_feature_pos, label_feature_neg, unlabel_feature_neg,name):
    #data, label, n_samples, n_features = get_data()
    label_feature_pos = np.array(label_feature_pos.cpu())
    unlabel_feature_pos = np.array(unlabel_feature_pos.cpu())
    label_feature_neg = np.array(label_feature_neg.cpu())
    unlabel_feature_neg = np.array(unlabel_feature_neg.cpu())

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    label_feature_pos_result = tsne.fit_transform(label_feature_pos)
    unlabel_feature_pos_result = tsne.fit_transform(unlabel_feature_pos)
    label_feature_neg_result = tsne.fit_transform(label_feature_neg)
    unlabel_feature_neg_result = tsne.fit_transform(unlabel_feature_neg)

    plot_embedding_labelunlabel_posneg(label_feature_pos_result, unlabel_feature_pos_result, label_feature_neg_result, unlabel_feature_neg_result,name)
    #plot_embedding_labelunlabel_posneg(label_feature_pos, unlabel_feature_pos, label_feature_neg, unlabel_feature_neg)

def feature_tsne_posneg_two(label_feature_pos, unlabel_feature_pos, name):
    #data, label, n_samples, n_features = get_data()
    label_feature_pos = np.array(label_feature_pos.cpu())
    unlabel_feature_pos = np.array(unlabel_feature_pos.cpu())
    lab_mean = np.mean(label_feature_pos)
    unlab_mean = np.mean(unlabel_feature_pos)
    lab_std = np.std(label_feature_pos)
    unlab_std = np.std(unlabel_feature_pos)

    label_feature_pos = (label_feature_pos-lab_mean)/lab_std
    unlabel_feature_pos = (unlabel_feature_pos-unlab_mean)/unlab_std
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    label_feature_pos_result = tsne.fit_transform(label_feature_pos)
    unlabel_feature_pos_result = tsne.fit_transform(unlabel_feature_pos)

    plot_embedding_labelunlabel_two(label_feature_pos_result, unlabel_feature_pos_result,name)
    #plot_embedding_labelunlabel_posneg(label_feature_pos, unlabel_feature_pos, label_feature_neg, unlabel_feature_neg)

def map_NCHW2NC(seg_feature,seg_label,channel,random_sample_ratio):
    seg_feature = seg_feature.detach()
    seg_label = seg_label.detach()
    seg_label[seg_label>0.5] = 1
    seg_label[seg_label<=0.5] = 0

    seg_label_pos_list = seg_label.permute(0, 2, 3, 1).reshape(-1, )
    seg_label_pos_list = seg_label_pos_list.bool()
    seg_feature_pos = seg_feature * seg_label

    seg_label_neg =  torch.logical_not(seg_label)
    seg_label_neg[seg_label_neg>0.5] = 1
    seg_label_neg[seg_label_neg<=0.5] = 0
    seg_label_neg_list = seg_label_neg.permute(0, 2, 3, 1).reshape(-1,)
    seg_label_neg_list = seg_label_neg_list.bool()
    seg_feature_neg = seg_feature * seg_label_neg

    seg_feature_neg = seg_feature_neg.permute(0, 2, 3, 1).reshape(-1, 64)
    seg_feature_pos = seg_feature_pos.permute(0, 2, 3, 1).reshape(-1, 64)
    seg_feature_neg_keep = seg_feature_neg[seg_label_neg_list]
    seg_feature_pos_keep = seg_feature_pos[seg_label_pos_list]


    resultList_neg = random.sample(range(0, seg_feature_neg_keep.shape[0]), int(seg_feature_neg_keep.shape[0]*random_sample_ratio*0.1*0.5))
    seg_feature_neg_keep_sample = seg_feature_neg_keep[resultList_neg]

    resultList_pos = random.sample(range(0, seg_feature_pos_keep.shape[0]), int(seg_feature_pos_keep.shape[0]*random_sample_ratio))
    seg_feature_pos_keep_sample = seg_feature_pos_keep[resultList_pos]


    return seg_feature_pos_keep_sample, seg_feature_neg_keep_sample



def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    print("data_shape",data.shape) #(N, 64)
    print("label_shape",label.shape) #(N, )
    print("label_shape",np.unique(label)) #(N, ) 0,1,2,3..for different classes : label pos and neg unlabel pos and neg
    result = tsne.fit_transform(data)
    plot_embedding(result, label,
                   't-SNE embedding of the digits (time %.2fs)'
                   % (time() - t0))


if __name__ == '__main__':
    # label_feature_pos = np.random.rand(100,128)
    # unlabel_feature_pos = np.random.rand(150,128)
    # label_feature_neg = np.random.rand(100,128)
    # unlabel_feature_neg = np.random.rand(150,128)
    # feature_tsne_posneg(label_feature_pos, unlabel_feature_pos, label_feature_neg, unlabel_feature_neg)
    # N 64 H W
    # N 1 H W

    # seg_feature = torch.rand(5,128,64,64)
    # seg_label = torch.rand(5,1,64,64)
    # #seg_label_neg = torch.rand(5, 1, 64, 64)
    #
    # seg_label[seg_label>0.5] = 1
    # seg_label[seg_label<=0.5] = 0
    #
    # seg_label_pos_list = seg_label.permute(0, 2, 3, 1).reshape(-1, )
    # seg_feature_pos = seg_feature * seg_label
    # seg_label_pos_list = seg_label_pos_list.bool()
    #
    # # seg_label_ = seg_label - 1
    # # seg_label_[seg_label_==0] = 1
    # # seg_label_[seg_label_<0] = 0
    # seg_label_neg =  torch.logical_not(seg_label)
    #
    # seg_label_neg[seg_label_neg>0.5] = 1
    # seg_label_neg[seg_label_neg <= 0.5] = 0
    # seg_label_neg_list = seg_label_neg.permute(0, 2, 3, 1).reshape(-1,)
    # seg_feature_neg = seg_feature * seg_label_neg
    # seg_label_neg_list = seg_label_neg_list.bool()
    #
    # seg_feature_neg = seg_feature_neg.permute(0, 2, 3, 1).reshape(-1, 128)
    # seg_feature_pos = seg_feature_pos.permute(0, 2, 3, 1).reshape(-1, 128)
    # print("seg_feature_pos",seg_feature_pos.shape)
    # print("seg_feature_neg",seg_feature_neg.shape)
    # print("seg_label_pos_list",seg_label_pos_list.shape)
    # print("seg_label_neg_list",seg_label_neg_list.shape)
    #
    # print("seg_label_pos_list_unique",torch.unique(seg_label_pos_list))
    # print("seg_label_neg_list_unique",torch.unique(seg_label_neg_list))
    # print("seg_label_pos_list",seg_label_pos_list.dtype)
    # print("seg_label_neg_list",seg_label_neg_list.dtype)
    # print("seg_label_neg_list",seg_label_neg_list.shape)
    # seg_feature_neg_keep = seg_feature_neg[seg_label_neg_list]
    # seg_feature_pos_keep = seg_feature_pos[seg_label_pos_list]
    # print("seg_feature_pos_keep",seg_feature_pos_keep.shape)
    # print("seg_feature_neg_keep",seg_feature_neg_keep.shape)
    #
    # next_obs_i = int(np.random.randint(0, seg_feature_neg_keep.shape[0]))
    # print("next_obs_i",next_obs_i)
    # resultList = random.sample(range(0, seg_feature_neg_keep.shape[0]), int(seg_feature_neg_keep.shape[0]*0.3))
    # print("len_result",len(resultList))
    # #print("len_result", resultList)
    # seg_feature_neg_keep_sample = seg_feature_neg_keep[resultList]
    # print("seg_feature_neg_keep_sample",seg_feature_neg_keep_sample.shape)

    # seg_feature_pos_keep = torch.rand(101,64)
    # seg_feature_pos_keep_2 = torch.rand(102,64)
    # car_feature = torch.vstack((seg_feature_pos_keep,seg_feature_pos_keep_2))
    # car_feature = torch.vstack((car_feature,seg_feature_pos_keep))
    # print("car_feature",car_feature.shape)

    #map_NCHW2NC(seg_feature, seg_label, random_sample_ratio)
    # seg_feature = torch.rand(5,128,64,64)
    # seg_label = torch.rand(5,1,64,64)
    # seg_label[seg_label>=0.5] = 1
    # seg_label[seg_label<0.5] = 0
    # random_sample_ratio = 0.3
    # seg_feature_pos_keep_sample, seg_feature_neg_keep_sample = map_NCHW2NC(seg_feature, seg_label, random_sample_ratio)
    # print("seg_feature_pos_keep_sample",seg_feature_pos_keep_sample.shape)
    # print("seg_feature_neg_keep_sample",seg_feature_neg_keep_sample.shape)

    seg_feature_pos_keep = torch.rand(101,64)
    seg_feature_pos_keep_2 = torch.rand(102,64)
    pos_list = []
    car_feature = torch.vstack((seg_feature_pos_keep,seg_feature_pos_keep_2))
    car_feature = torch.vstack((car_feature,seg_feature_pos_keep))
    pos_list.append(seg_feature_pos_keep)
    pos_list.append(seg_feature_pos_keep_2)
    arr0 = pos_list[0]
    for i in range(1, len(pos_list)):
        arr0 = torch.vstack((arr0,pos_list[i]))
    print("arr0",arr0.shape)








