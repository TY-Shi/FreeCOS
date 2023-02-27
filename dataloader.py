import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape
from datasets.BaseDataset import BaseDataset
from utils.LIOT_torch import trans_liot
from utils.FDA import FDA_source_to_target

def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def random_mirror(img, gt=None):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)

    return img, gt

def random_scale(img, gt=None, scales=None):
    scale = random.choice(scales)
    # scale = random.uniform(scales[0], scales[-1])
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if gt is not None:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale

def SemanticEdgeDetector(gt):
    id255 = np.where(gt == 255)
    no255_gt = np.array(gt)
    no255_gt[id255] = 0
    cgt = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
    edge_radius = 7
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_radius, edge_radius))
    cgt = cv2.dilate(cgt, edge_kernel)
    # print(cgt.max(), cgt.min())
    cgt[cgt>0] = 1
    return cgt

class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt=None):
        # gt = gt - 1     # label 0 is invalid, this operation transfers label 0 to label 255
        # random mirror
        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)#random scale

        img = normalize(img, self.img_mean, self.img_std)
        # if gt is not None:
        #     cgt = SemanticEdgeDetector(gt)#canny gt
        # else:
        #     cgt = None

        crop_size = (config.image_height, config.image_width)#256,256
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
            # p_cgt, _ = random_crop_pad_to_shape(cgt, crop_pos, crop_size, 255)
        else:
            p_gt = None
            # p_cgt = None

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = {}

        return p_img, p_gt, extra_dict

class TrainvalPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt=None):
        # gt = gt - 1     # label 0 is invalid, this operation transfers label 0 to label 255

        img = normalize(img, self.img_mean, self.img_std)
        p_img = img.transpose(2, 0, 1)
        extra_dict = {}

        return p_img, gt, None, extra_dict

class ValPre(object):
    def __call__(self, img, gt):
        # gt = gt - 1
        extra_dict = {}
        return img, gt, None, extra_dict


def get_train_loader(engine, dataset, train_source, unsupervised=False):
    data_setting = {'img_root': config.train_img_root_folder, #/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/img
                    'gt_root': config.train_fakegt_root_folder, #/mnt/nas/sty/codes/Unsupervised/Data/XCAD/train/fake_vessel
                    'train_source': train_source, #/mnt/nas/sty/codes/Unsupervised/Data/XCAD/config_new/subset_train/train_aug_labeled_1-{}.txt
                    'eval_source': config.eval_source}#/mnt/nas/sty/codes/Unsupervised/Data/XCAD/config_new/val.txt
    train_preprocess = TrainPre(config.image_mean, config.image_std)#mean:np.array([0.485, 0.456, 0.406]) std:np.array([0.229, 0.224, 0.225])

    if unsupervised is False:
        train_dataset = dataset(data_setting, "train", train_preprocess,
                                config.max_samples, unsupervised=False)
    else:
        train_dataset = dataset(data_setting, "train", train_preprocess,
                                config.max_samples, unsupervised=True)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

def get_val_loader(engine, dataset, train_source, unsupervised=False):
    data_setting = {'img_root': config.img_root_folder, #/mnt/nas/sty/codes/Unsupervised/Data/XCAD
                    'gt_root': config.gt_root_folder, #/mnt/nas/sty/codes/Unsupervised/Data/XCAD
                    'train_source': train_source, #/mnt/nas/sty/codes/Unsupervised/Data/XCAD/config_new/subset_train/train_aug_labeled_1-{}.txt
                    'eval_source': config.eval_source}#/mnt/nas/sty/codes/Unsupervised/Data/XCAD/config_new/val.txt
    train_preprocess = TrainvalPre(config.image_mean, config.image_std)#mean:np.array([0.485, 0.456, 0.406]) std:np.array([0.229, 0.224, 0.225])


    train_dataset = dataset(data_setting, "trainval", train_preprocess,
                            config.max_samples, unsupervised=False)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

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

class XCAD(BaseDataset):
    trans_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 31, 32, 33]
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, training=True, unsupervised=False, lioted=False):
        super(XCAD, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess# It is a image propocess model
        self.training = training
        self.unsupervised = unsupervised
        self.liot = lioted

    def __getitem__(self, index):
        if not self.unsupervised:#with gt need FDA make new image with fakevessel and
            idx_img = np.random.randint(self._file_length)
            if self._file_length is not None:
                gt_names = self._construct_new_file_names(self._file_length)[index]
                img_names = self._construct_new_file_names(self._file_length)[idx_img]
            else:
                gt_names = self._file_names[index]
                img_names = self._file_names[idx_img]
        else:
            if self._file_length is not None:
                img_names = self._construct_new_file_names(self._file_length)[index]
                gt_names = img_names
            else:
                img_names = self._file_names[index]
                gt_names = img_names
        img_path = self._img_path+img_names[0] #os.path.join(self._img_path, names[0])
        gt_path = self._gt_path+gt_names[1] #os.path.join(self._gt_path, names[1])
        item_name = gt_names[1].split("/")[-1].split(".")[0]

        # print(img_path)

        if not self.unsupervised:
            img, gt = self._fetch_data(img_path, gt_path)
        else:
            #need random fakevessel and img_back
            img_back, gt_fakevessel = self._fetch_data(img_path, gt_path)
            gt = None
            gt_revser = replace_color(gt_fakevessel)
            img = FDA_source_to_target(img_back, gt_revser, L=0.1)


        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, edge_gt, extra_dict = self.preprocess(img, gt)

        if self._split_name in ['train', 'trainval', 'train_aug', 'trainval_aug']:
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            if gt is not None:
                gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
                #edge_gt = torch.from_numpy(np.ascontiguousarray(edge_gt)).long()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()
        if self.liot:
            img = trans_liot(img)
        output_dict = dict(data=img, fn=str(item_name),
                           n=len(self._file_names))
        if gt is not None:
            extra_dict['label'] = gt

        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path=None, dtype=None):
        img = self._open_image(img_path)
        if gt_path is not None:
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
            return img, gt
        return img, None


class CityScape(BaseDataset):
    trans_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 31, 32, 33]
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, training=True, unsupervised=False):
        super(CityScape, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.training = training
        self.unsupervised = unsupervised

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = self._img_path+names[0] #os.path.join(self._img_path, names[0])
        gt_path = self._gt_path+names[1] #os.path.join(self._gt_path, names[1])
        item_name = names[1].split("/")[-1].split(".")[0]

        # print(img_path)

        if not self.unsupervised:
            img, gt = self._fetch_data(img_path, gt_path)
        else:
            img, gt = self._fetch_data(img_path, None)

        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, edge_gt, extra_dict = self.preprocess(img, gt)

        if self._split_name in ['train', 'trainval', 'train_aug', 'trainval_aug']:
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            if gt is not None:
                gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
                edge_gt = torch.from_numpy(np.ascontiguousarray(edge_gt)).long()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, fn=str(item_name),
                           n=len(self._file_names))
        if gt is not None:
            extra_dict['label'] = gt

        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path=None, dtype=None):
        img = self._open_image(img_path)
        if gt_path is not None:
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
            return img, gt
        return img, None

    @classmethod
    def get_class_colors(*args):
        return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]

    @classmethod
    def get_class_names(*args):
        return ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name
