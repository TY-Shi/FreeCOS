import random
import time

import torch
import os.path
import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as F

from torchvision import transforms
from torch.utils import data
from Datasetloader.torch_LIOT import trans_liot
import cv2
from Datasetloader.elastic_transform import elastic_transform_PIL
from Datasetloader.torch_LIOT import trans_liot, trans_liot_region, trans_liot_region_stride, trans_liot_differentsize
#trans_list NCHW

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


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    """NCHW"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    n, c, h, w = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12

class DatasetSTARE_aug(data.Dataset):
    def __init__(self, benchmark, datapath, split, img_mode, img_size,supervised):
        super(DatasetSTARE_aug, self).__init__()
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.benchmark = benchmark
        assert self.benchmark == 'STARE_LIOT'
        self.img_mode = img_mode
        assert img_mode in ['crop', 'same', 'resize']
        self.img_size = img_size
        self.supervised = supervised

        self.datapath = datapath
        if self.supervised=='supervised':
            if self.split == 'train':
                self.img_path = os.path.join(datapath, 'train','fake_vessel_more_thinthick')#fake vessel gray img
                self.background_path = os.path.join(datapath,'train', 'img')#target domain img
                self.ann_path = os.path.join(datapath, 'train','fake_vessel_gt_thinthick')#fake vessel gt
                self.ignore_path = os.path.join(datapath, 'train','mask')#fake vessel gt
                self.img_metadata = self.load_metadata_supervised() #train_fakevessel.txt
                self.background_metadata = self.load_metadata_background()#train_background.txt[0,1,2,3,4]
            else:
                self.img_path = os.path.join(datapath, 'test','img')
                self.ann_path = os.path.join(datapath, 'test','gt')
                self.ignore_path = os.path.join(datapath, 'test','mask')#fake vessel gt
                self.img_metadata = self.load_metadata_testsupervised()#test_img.txt
        else:
            self.img_path = os.path.join(datapath, 'train','img')
            self.ignore_path = os.path.join(datapath, 'train', 'mask')  #fake vessel gt
            self.img_metadata = self.load_metadata_background()#train_background.txt
        self.norm_img = transforms.Compose([
            transforms.ToTensor()
        ])
        if self.img_mode == 'resize':
            self.resize = transforms.Resize([img_size, img_size], interpolation=Image.NEAREST)
        else:
            self.resize = None

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, index):
        img_name = self.img_metadata[index]#get the name of fakevessel(train-supervised), img(test-supervised), img(train-unsupervised)
        if self.supervised=='supervised' and self.split == 'train':
            idx_background = np.random.randint(len(self.background_metadata))  # background(train-supervised)
            background_name = self.background_metadata[idx_background]
            img, anno_mask, background_img, org_img_size = self.load_frame_fakevessel_whole_center(img_name, background_name)
            ignore_mask = None
        elif self.supervised=='supervised' and self.split != 'train':
            img, anno_mask, ignore_mask, org_img_size = self.load_frame_aff_mask(img_name)
        else:
            img, ignore_mask, org_img_size = self.load_frame_unsupervised_ignoremask(img_name)
            anno_mask = None
            background_img = None

        if self.split == 'train' and self.supervised!='supervised':
            img, anno_mask = self.augmentation_unsupervised(img,anno_mask)

        if self.img_mode == 'resize' and self.split == 'train' :
            img = self.resize(img)
            if anno_mask!=None:
                anno_mask = self.resize(anno_mask)
        elif self.img_mode == 'crop' and self.split == 'train' and self.supervised!='supervised':
            if background_img!=None:
                i, j, background_img, h, w = self.get_params_center(img, background_img, (self.img_size, self.img_size))
                img = F.crop(img, i, j, h, w)
            else:
                i, j, h, w = self.get_params(img, (self.img_size, self.img_size))
                img = F.crop(img, i, j, h, w)
                if ignore_mask!=None:
                    #print("unsupervised_orgin_mask",ignore_mask.shape)
                    i_g, j_g, h_g, w_g = self.get_params(ignore_mask, (self.img_size, self.img_size))
                    ignore_mask = F.crop(ignore_mask, i_g, j_g, h_g, w_g)
                    #print("unsupervised_mask",ignore_mask.shape)
                if anno_mask!=None:
                    anno_mask = F.crop(anno_mask, i, j, h, w)
        else:
            pass
        img_array = np.array(img)
        img_array = trans_liot(img_array)
        img_array = img_array.transpose((1, 2, 0))
        img = self.norm_img(img_array)

        if self.supervised=='supervised':
            if ignore_mask==None:
                ignore_mask=0
            batch = {
                'img_name': img_name,
                'img': img,
                'anno_mask': anno_mask,
                'ignore_mask':ignore_mask
            }
            return batch
        else:
            batch = {
                'img_name': img_name,
                'img': img
            }
            return batch

    def augmentation(self, img, anno_mask, anno_boundary, ignore_mask):
        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)
        img = transform_hflip(img)
        anno_mask = transform_hflip(anno_mask)
        anno_boundary = transform_hflip(anno_boundary)
        ignore_mask = transform_hflip(ignore_mask)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)
        img = transform_vflip(img)
        anno_mask = transform_vflip(anno_mask)
        anno_boundary = transform_vflip(anno_boundary)
        ignore_mask = transform_vflip(ignore_mask)

        #img = self.to_tensor(img)
        if np.random.random() > 0.5:
            p = np.random.uniform(-180,180,1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)
            anno_mask = transform_rotate(anno_mask)
            anno_boundary = transform_rotate(anno_boundary)
            ignore_mask = transform_rotate(ignore_mask)

        if np.random.random() > 0.5:
            color_aug = transforms.ColorJitter(brightness=[2.1, 2.1], contrast=[2.1, 2.1], saturation=[1.5, 1.5])
            img = color_aug(img)

        return img, anno_mask, anno_boundary, ignore_mask

    def augmentation_aff(self, img, anno_mask):
        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)
        img = transform_hflip(img)
        anno_mask = transform_hflip(anno_mask)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)
        img = transform_vflip(img)
        anno_mask = transform_vflip(anno_mask)
        if np.random.random() > 0.5:
            p = np.random.uniform(-180,180,1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)
            anno_mask = transform_rotate(anno_mask)

        if np.random.random() > 0:
            color_aug = transforms.ColorJitter(brightness=[1.0, 1.3], contrast=[0.2, 0.6], saturation=[0.3, 0.5])#augLOwcon0.8-1.5、0.5-1.5、0.5-2.1 brigntness 0.5-1
            img = color_aug(img)
        transform = transforms.CenterCrop((self.img_size, self.img_size))
        anno_mask = transform(anno_mask)
        img = transform(img)

        return img, anno_mask

    def augmentation_unsupervised(self, img, anno_mask):
        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)
        img = transform_hflip(img)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)
        img = transform_vflip(img)

        if np.random.random() > 0.5:
            p = np.random.uniform(-180,180,1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)

        if np.random.random() > 0.5:
            color_aug = transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.8, 2.1], saturation=[0.5, 1.5])#augLOwcon0.8-1.5、0.5-1.5、0.5-2.1 brigntness 0.5-1
            img = color_aug(img)
        return img, anno_mask

    def load_frame(self, img_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        anno_boundary = self.read_boundary(img_name)
        ignore_mask = self.read_ignore_mask(img_name)

        org_img_size = img.size

        return img, anno_mask, anno_boundary, ignore_mask, org_img_size

    def load_frame_aff(self, img_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        org_img_size = img.size

        return img, anno_mask, org_img_size

    def load_frame_aff_mask(self, img_name):
        img = self.read_img(img_name)
        anno_mask = self.read_testmask(img_name)
        ignore_mask = self.read_ignore_mask_torch(img_name)

        org_img_size = img.size

        return img, anno_mask, ignore_mask, org_img_size

    def load_frame_fakevessel(self,img_name,background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)

        background_array = np.array(background_img)
        im_src = np.asarray(img, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src,axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.02)
        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA,axis = 0)
        img_FDA_Image = Image.fromarray((img_FDA).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_fakevessel_gaussian(self,img_name,background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)

        background_array = np.array(background_img)
        im_src = np.asarray(img, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src,axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.3)

        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA,axis = 0)
        #guassian_blur
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (13, 13), 0)
        noise_map = np.random.uniform(-5,5,img_FDA_guassian.shape)
        img_FDA_guassian = img_FDA_guassian + noise_map
        img_FDA_guassian = np.clip(img_FDA_guassian, 0, 255.)

        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, ignore_mask, org_img_size

    def load_frame_fakevessel_whole(self,img_name,background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)
        ignore_mask = self.read_ignore_mask(background_name)
        org_img_size = img.size

        return img, anno_mask, background_img, ignore_mask, org_img_size

    def load_frame_fakevessel_whole_center(self,img_name,background_name):
        img = self.read_img(img_name)#PIL
        anno_mask = self.read_mask(img_name)#torch
        background_img = self.read_background(background_name)#PIL
        org_img_size = img.size
        img_crop,bakground_crop, anno_crop = self.get_params_center_forvessel(img, background_img, anno_mask, (self.img_size, self.img_size))
        background_array = np.array(bakground_crop)
        im_src = np.asarray(img_crop, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src,axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.3)
        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA,axis = 2)
        #guassian_blur
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (7, 7), 0)
        noise_map = np.random.uniform(-10,10,img_FDA_guassian.shape)
        img_FDA_guassian = img_FDA_guassian + noise_map
        img_FDA_guassian = np.clip(img_FDA_guassian, 0, 255.)
        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')
        img_FDA_Image, anno_crop = self.augmentation_aff(img_FDA_Image, anno_crop)
        return img_FDA_Image, anno_crop, bakground_crop, org_img_size

    def image_trans_guassainliot(self,img, background_img):
        background_array = np.array(background_img)
        im_src = np.asarray(img, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src,axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.3)

        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA,axis = 0)
        #guassian_blur
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (13, 13), 0)
        noise_map = np.random.uniform(-5,5,img_FDA_guassian.shape)
        img_FDA_guassian = img_FDA_guassian + noise_map
        img_FDA_guassian = np.clip(img_FDA_guassian, 0, 255.)
        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')
        return img_FDA_Image


    def load_frame_fakevessel_gaussian_intensity(self,img_name,background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)

        background_array = np.array(background_img)
        gt_arrray = np.array(anno_mask)
        im_src = np.asarray(img, np.float32)
        im_src[gt_arrray[0,:,:]==1] = 220 #change the src intensity
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src,axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))

        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.02)

        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA,axis = 0)
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (13, 13), 0)

        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_fakevessel_elastic(self,img_name,background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)
        gt_array = np.array(anno_mask)#tensor tp array
        gt_mask = np.squeeze(gt_array, axis=0)*255
        background_array = np.array(background_img)
        im_src = np.asarray(img, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src,axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.02)

        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA,axis = 0)
        img_FDA_Image = Image.fromarray((img_FDA).astype('uint8')).convert('L')
        gt_Image = Image.fromarray((gt_mask).astype('uint8')).convert('L')
        image_deformed, mask_deformed = elastic_transform_PIL(img_FDA_Image, gt_Image, img_FDA.shape[1] * 2,img_FDA.shape[1] * 0.2, img_FDA.shape[1] * 0.1)
        mask_deformed[mask_deformed == 0] = 0
        mask_deformed[mask_deformed == 255] = 1
        mask_deformed = torch.from_numpy(mask_deformed).float().unsqueeze(0)
        img_deform_Image = Image.fromarray((image_deformed).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_deform_Image, mask_deformed, org_img_size

    def load_frame_fakevessel_cutvessel(self,img_name,background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)
        im_array = np.array(img)
        gt_arrray = np.array(anno_mask)

        background_array = np.array(background_img)
        img_FDA_r = np.where(gt_arrray[0,:,:]>0,im_array ,background_array)
        img_FDA_Image = Image.fromarray((img_FDA_r).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_unsupervised(self, img_name):
        img = self.read_img(img_name)
        org_img_size = img.size
        return img, org_img_size

    def load_frame_unsupervised_ignoremask(self, img_name):
        img = self.read_img(img_name)
        ignore_mask = None
        org_img_size = img.size
        return img, ignore_mask, org_img_size

    def load_frame_supervised(self, img_name, idx_background):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)

        org_img_size = img.size

        return img, anno_mask, org_img_size

    def read_mask(self, img_name):
        gt_name = img_name.split(".")[0]+'.png'
        mask = np.array(Image.open(os.path.join(self.ann_path, gt_name)).convert('L'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_testmask(self, img_name):
        gt_name = img_name.split(".")[0]+'.ah.ppm'
        mask = np.array(Image.open(os.path.join(self.ann_path, gt_name)).convert('L'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask


    def read_ignore_mask(self, img_name):
        mask_name = img_name.split(".")[0]
        mask = Image.open(os.path.join(self.ignore_path, mask_name) + '.png').convert("L")
        return mask


    def read_ignore_mask_torch(self, img_name):
        mask_name = img_name.split(".")[0]
        mask = np.array(Image.open(os.path.join(self.ignore_path, mask_name) + '.ppm'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_boundary(self, img_name):
        mask = np.array(Image.open(os.path.join(self.bd_path, img_name) + '.png'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_img(self, img_name):
        # maybe png
        time.sleep(0.001)
        RGB_Image = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        RGB_array = np.array(RGB_Image)
        Gray_array = RGB_array[:,:,1]
        Gray_Image =Image.fromarray(Gray_array)
        return Gray_Image

    def read_background(self,img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')

    def load_metadata(self):
        if self.split == 'train':
            meta_file = os.path.join(self.datapath, 'split', 'train.txt')
        elif self.split == 'val' or self.split == 'test':
            meta_file = os.path.join(self.datapath, 'split', 'test.txt')
        else:
            raise RuntimeError('Undefined split ', self.split)

        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    def load_metadata_supervised(self):
        # train_fakevessel.txt
        if self.split == 'train':
            meta_file = os.path.join(self.datapath, 'split', 'train_fakevessel.txt')

        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    def load_metadata_background(self):
        # train_backvessel.txt
        if self.split == 'train':
            meta_file = os.path.join(self.datapath, 'split', 'train_background.txt')
        print("unsupervised_metafile:",meta_file)
        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname
    #load_metadata_testsupervised
    def load_metadata_testsupervised(self):
        # test_img.txt
        if self.split == 'test' or 'val':
            meta_file = os.path.join(self.datapath, 'split', 'test_img.txt')
        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    def get_params(self, img, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th-1)
        j = random.randint(0, w - tw-1)
        return i, j, th, tw

    def get_params_center(self, img, background_mask, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th-1)
        j = random.randint(0, w - tw-1)
        transform = transforms.CenterCrop((th, tw))
        background_mask = transform(background_mask)
        return i, j, background_mask, th, tw

    def get_params_center_forvessel(self, img, background_img, anno_mask, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                #print("tensor_image",img.shape)
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th-1)
        j = random.randint(0, w - tw-1)
        img_crop = F.crop(img, i, j, th, tw)
        anno_mask_crop = F.crop(anno_mask,i,j,th,tw)
        transform = transforms.RandomCrop((th, tw))
        background_img = transform(background_img)
        return img_crop,background_img,anno_mask_crop