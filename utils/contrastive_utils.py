import numpy as np
import torch
import random
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn.functional as F
import PIL.Image as Image

random.seed(1)


def enhance_op(x):
    assert x.dim() == 4, f'only support BxCxHxW which dim=4'
    N, C, H, W = x.shape
    phi_x = F.normalize(x, p=2, dim=1, eps=1e-12, out=None).view(N, C, -1)
    theta_x = F.normalize(x, p=2, dim=1, eps=1e-12, out=None).view(N, C, -1).permute(0, 2, 1)
    pairwise_weight = torch.matmul(theta_x, phi_x)
    pairwise_weight = pairwise_weight.softmax(dim=-1)
    x = x.view(N, C, -1).permute(0, 2, 1)
    out = torch.matmul(pairwise_weight, x)
    out = out.permute(0, 2, 1).contiguous().reshape(N, C, H, W)

    return out


def normalize_batch(cams_batch):
    """
    Classic min-max normalization
    """
    bs = cams_batch.size(0)
    cams_batch = cams_batch + 1e-4
    cam_mins = getattr(cams_batch.view(bs, -1).min(1), 'values').view(bs, 1, 1, 1)
    cam_maxs = getattr(cams_batch.view(bs, -1).max(1), 'values').view(bs, 1, 1, 1)
    return (cams_batch - cam_mins) / (cam_maxs - cam_mins)


def get_query_keys_eval(cams):
    """
        Input
            cams: Tensor, cuda, Nx1x28x28

        Here, when performing evaluation, only cams are provided for all categories, including base and novel.
    """
    return_result = dict()
    cams = cams.squeeze(1).cpu()
    cams = normalize_zero_to_one(cams)  # tensor  shape:N,28,28, 0-1

    # we only need queries
    query_pos_sets = torch.where(cams > 0.92, 1.0, 0.0).to(dtype=torch.bool)
    query_neg_sets = torch.where(cams < 0.08, 1.0, 0.0).to(dtype=torch.bool)

    return_result['query_pos_sets'] = query_pos_sets
    return_result['query_neg_sets'] = query_neg_sets

    return return_result


def get_query_keys(
        cams,
        edges,
        masks=None,
        is_novel=None,
        thred_u=0.1,
        scale_u=1.0,
        percent=0.3):
    """
        Input
            cams: Tensor, cuda, Nx1x28x28
            edges: Tensor, cuda, Nx28x28
            masks: Tensor, cuda, Nx28x28
    """
    #######################################################
    # ---------- some pre-processing -----------------------
    #######################################################
    cams = cams.squeeze(1).cpu()  # to cpu, Nx28x28
    edges = edges.cpu()  # to cpu, Nx28x28
    masks = masks.cpu()  # to cpu, Nx28x28
    cams = normalize_zero_to_one(cams)  # normalize to 0~1

    #######################################################
    # ---------- get query mask for each proposal ----------
    #######################################################
    query_pos_sets = masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
    query_neg_sets = torch.logical_not(query_pos_sets)

    # if available points in a mask less than 2. replace it with cam. Note cam is available for base(seen) and novel(unseen)
    keep_pos_flag = torch.where(query_pos_sets.sum(dim=[1, 2]) < 2.0, 1.0, 0.0).to(dtype=torch.bool)
    keep_neg_flag = torch.where(query_neg_sets.sum(dim=[1, 2]) < 2.0, 1.0, 0.0).to(dtype=torch.bool)
    if True in keep_pos_flag:
        cam_pos_high = torch.where(cams > 0.95, 1.0, 0.0).to(dtype=torch.bool)
        query_pos_sets[keep_pos_flag] = cam_pos_high[keep_pos_flag]  # replace pos-mask with high-confidence (0.95) cam
    if True in keep_neg_flag:
        cam_neg_high = torch.where(cams < 0.05, 1.0, 0.0).to(dtype=torch.bool)
        query_neg_sets[keep_neg_flag] = cam_neg_high[keep_neg_flag]  # replace neg-mask with low-confidence (0.05) cam

    # For novel(unseen), replace query mask via cam's confidence since query mask are unknown for novel(unseen)
    unseen_query_pos_sets = torch.where(cams > (1 - thred_u), 1.0, 0.0).to(dtype=torch.bool)
    unseen_query_neg_sets = torch.where(cams < thred_u, 1.0, 0.0).to(dtype=torch.bool)
    query_pos_sets[is_novel] = unseen_query_pos_sets[is_novel]
    query_neg_sets[is_novel] = unseen_query_neg_sets[is_novel]

    #######################################################
    # ----------- get different types of keys -------------
    #######################################################
    # For base(seen), get keys according to gt_mask and edges
    edge_sets_dilate = get_pixel_sets_distrans(edges, radius=2)  # expand edges with radius=2
    hard_pos_neg_sets = edge_sets_dilate - edges  # hard keys for both pos and neg

    # different sets, you can refer to the figure in https://blog.huiserwang.site/2022-03/Project-ContrastMask/ to easily understand.
    hard_negative_sets = torch.where((hard_pos_neg_sets - masks) > 0.5, 1.0, 0.0)
    hard_positive_sets = torch.where((hard_pos_neg_sets - hard_negative_sets) > 0.5, 1.0, 0.0)
    easy_positive_sets = torch.where((masks - hard_pos_neg_sets) > 0.5, 1.0, 0.0)
    easy_negative_sets = torch.logical_not(torch.where((masks + edge_sets_dilate) > 0.5, 1.0, 0.0)).to(
        dtype=easy_positive_sets.dtype)

    # for novel(unseen), get keys according to cam, hard and easy are both sampled in the same sets, replace original sets
    unseen_positive_sets = torch.where(cams > (1.0 - thred_u * scale_u), 1.0,
                                       0.0)  # scale_u can adjust the threshold, it is not used in our paper.
    unseen_negative_sets = torch.where(cams < (thred_u * scale_u), 1.0, 0.0)
    easy_positive_sets[is_novel] = unseen_positive_sets[is_novel]
    easy_negative_sets[is_novel] = unseen_negative_sets[is_novel]
    hard_positive_sets[is_novel] = unseen_positive_sets[is_novel]
    hard_negative_sets[is_novel] = unseen_negative_sets[is_novel]

    #######################################################
    # --------- determine the number of sampling ----------
    #######################################################
    # how many points can be sampled for all proposals for each type of sets
    num_Epos_ = easy_positive_sets.sum(dim=[1, 2])  # E=easy, H=hard
    num_Hpos_ = hard_positive_sets.sum(dim=[1, 2])
    num_Eneg_ = easy_negative_sets.sum(dim=[1, 2])
    num_Hneg_ = hard_negative_sets.sum(dim=[1, 2])

    # if available points are less then 5 for each type, this proposal will be dropped out.
    available_num = torch.cat([num_Epos_, num_Eneg_, num_Hpos_, num_Hneg_])
    abandon_inds = torch.where(available_num < 5, 1, 0).reshape(4, -1)
    keeps = torch.logical_not(abandon_inds.sum(0).to(dtype=torch.bool))
    if True not in keeps:  # all proposals do not have enough points that can be sample. This is a extreme situation.
        # set the points number of all types sets to 2
        # sometimes, there would still raise an error. I will fix it later.
        sample_num_Epos = torch.ones_like(num_Epos_) * 2
        sample_num_Hpos = torch.ones_like(num_Hpos_) * 2
        sample_num_Eneg = torch.ones_like(num_Eneg_) * 2
        sample_num_Hneg = torch.ones_like(num_Hneg_) * 2
        # print('[sample points]:{}'.format(available_num))  # print log so that we can debug it.....
    else:
        sample_num_Epos = (percent * num_Epos_[keeps]).ceil()  # percent is the sigma in our paper
        sample_num_Hpos = (percent * num_Hpos_[keeps]).ceil()
        sample_num_Eneg = (percent * num_Eneg_[keeps]).ceil()
        sample_num_Hneg = (percent * num_Hneg_[keeps]).ceil()

    #######################################################
    # ----------------- sample points ---------------------
    #######################################################
    easy_positive_sets_N = get_pixel_sets_N(easy_positive_sets[keeps], sample_num_Epos)
    easy_negative_sets_N = get_pixel_sets_N(easy_negative_sets[keeps], sample_num_Eneg)
    hard_positive_sets_N = get_pixel_sets_N(hard_positive_sets[keeps], sample_num_Hpos)
    hard_negative_sets_N = get_pixel_sets_N(hard_negative_sets[keeps], sample_num_Hneg)

    # Record points number
    num_per_type = dict()
    num_per_type['Epos_num_'] = sample_num_Epos
    num_per_type['Hpos_num_'] = sample_num_Hpos
    num_per_type['Eneg_num_'] = sample_num_Eneg
    num_per_type['Hneg_num_'] = sample_num_Hneg

    #######################################################
    # ------------------- return data ---------------------
    #######################################################
    return_result = dict()
    return_result['keeps'] = keeps  # which proposal is preserved
    return_result['num_per_type'] = num_per_type
    return_result['query_pos_sets'] = query_pos_sets  # query area for foreground
    return_result['query_neg_sets'] = query_neg_sets  # query area for background
    return_result['easy_positive_sets_N'] = easy_positive_sets_N.to(dtype=torch.bool)
    return_result['easy_negative_sets_N'] = easy_negative_sets_N.to(dtype=torch.bool)
    return_result['hard_positive_sets_N'] = hard_positive_sets_N.to(dtype=torch.bool)
    return_result['hard_negative_sets_N'] = hard_negative_sets_N.to(dtype=torch.bool)
    return return_result


def get_query_keys_sty(
        edges,
        masks=None,
        thred_u=0.1,
        scale_u=1.0,
        percent=0.3,
        fake=True):
    """
        Input
            edges: Tensor, cuda, Nx28x28
            masks: Tensor, cuda, Nx28x28 is the mask or prediction 0-1 0r [0,1]
    """
    #######################################################
    # ---------- some pre-processing -----------------------
    #######################################################
    masks = masks.cpu()  # to cpu, Nx28x28
    #######################################################
    # ---------- get query mask for each proposal ----------#
    #######################################################
    if fake:
        query_pos_sets = masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = torch.logical_not(query_pos_sets)
        edges = edges.cpu()  # to cpu, Nx28x28
        # print("Real_query_pos_sets", query_pos_sets.shape)
    else:
        pos_masks = torch.where(masks > (1 - thred_u), 1.0, 0.0).to(dtype=torch.bool)
        neg_mask = torch.where(masks < thred_u, 1.0, 0.0).to(dtype=torch.bool)
        query_pos_sets = pos_masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = neg_mask.to(dtype=torch.bool)
        # edges = edges.cpu()  # to cpu, Nx28x28
        # print("edges",edges.shape)# 8 1 256 256
        # print("query_pos_sets", query_pos_sets.shape)# 8 1 256 256

    # if available points in a mask less than 2. replace it with cam. Note cam is available for base(seen) and novel(unseen)
    # # For novel(unseen), replace query mask via cam's confidence since query mask are unknown for novel(unseen)
    #######################################################
    # ----------- get different types of keys -------------
    #######################################################
    # For base(seen), get keys according to gt_mask and edges
    # edge_sets_dilate = get_pixel_sets_distrans(edges, radius=2)  # expand edges with radius=2
    # hard_pos_neg_sets = edge_sets_dilate - edges  # hard keys for both pos and neg
    # different sets, you can refer to the figure in https://blog.huiserwang.site/2022-03/Project-ContrastMask/ to easily understand.
    if fake:
        edge_sets_dilate = get_pixel_sets_distrans(edges, radius=2)  # expand edges with radius=2
        hard_pos_neg_sets = edge_sets_dilate - edges  # hard keys for both pos and neg
        hard_negative_sets = torch.where((hard_pos_neg_sets - masks) > 0.5, 1.0, 0.0)
        # hard_positive_sets = torch.where((hard_pos_neg_sets - hard_negative_sets) > 0.5, 1.0, 0.0)
        hard_positive_sets = torch.where((hard_pos_neg_sets - hard_negative_sets + edges) > 0.5, 1.0,
                                         0.0)  # include edage
        easy_positive_sets = torch.where((masks - hard_pos_neg_sets) > 0.5, 1.0, 0.0)
        easy_negative_sets = torch.logical_not(torch.where((masks + edge_sets_dilate) > 0.5, 1.0, 0.0)).to(
            dtype=easy_positive_sets.dtype)
        # print("easy_negative_sets",easy_negative_sets.shape)#8 1 256 256
        # print("easy_positive_sets", easy_positive_sets.shape)#8 1 256 256
        # print("hard_positive_sets", hard_positive_sets.shape)#8 1 256 256
        # print("hard_negative_sets", hard_negative_sets.shape)#8 1 256 256
        # print("hard_pos_neg_sets", hard_pos_neg_sets.shape)#8 1 256 256
    else:
        # for novel(unseen), get keys according to cam, hard and easy are both sampled in the same sets, replace original sets
        unseen_positive_sets = torch.where(masks > (1.0 - thred_u * scale_u), 1.0,
                                           0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        unseen_negative_sets = torch.where(masks < (thred_u * scale_u), 1.0, 0.0)
        easy_positive_sets = unseen_positive_sets
        easy_negative_sets = unseen_negative_sets
        hard_positive_sets = unseen_positive_sets
        hard_negative_sets = unseen_negative_sets
        # print("ReaL_easy_negative_sets",easy_negative_sets.shape)
        # print("Real_easy_positive_sets", easy_positive_sets.shape)
        # print("Real_hard_positive_sets", hard_positive_sets.shape)
        # print("Real_hard_negative_sets", hard_negative_sets.shape)
    #######################################################
    # --------- determine the number of sampling ----------
    #######################################################
    # how many points can be sampled for all proposals for each type of sets
    num_Epos_ = easy_positive_sets.sum(dim=[2, 3])  # E=easy, H=hard
    num_Hpos_ = hard_positive_sets.sum(dim=[2, 3])
    num_Eneg_ = easy_negative_sets.sum(dim=[2, 3])
    num_Hneg_ = hard_negative_sets.sum(dim=[2, 3])
    # print("num_Epos_",num_Epos_)#
    # print("num_Hpos_",num_Hpos_)
    # print("num_Eneg_",num_Eneg_)
    # print("num_Hneg_",num_Hneg_)
    # print("num_Hneg_",num_Hneg_.shape)#8 1
    # if available points are less then 5 for each type, this proposal will be dropped out.
    available_num = torch.cat([num_Epos_, num_Eneg_, num_Hpos_, num_Hneg_])
    # print("available_num",available_num.shape)#32 , 1
    abandon_inds = torch.where(available_num < 5, 1, 0).reshape(4, -1)
    # print("abandon_inds", abandon_inds.shape)#4 , 8
    keeps = torch.logical_not(abandon_inds.sum(0).to(dtype=torch.bool))
    # print("keeps",keeps.shape)#8
    # print("keeps_show",keeps)#[T,F]
    if True not in keeps:  # all proposals do not have enough points that can be sample. This is a extreme situation.
        # set the points number of all types sets to 2
        # sometimes, there would still raise an error. I will fix it later.
        sample_num_Epos = torch.ones_like(num_Epos_) * 2
        sample_num_Hpos = torch.ones_like(num_Hpos_) * 2
        sample_num_Eneg = torch.ones_like(num_Eneg_) * 2
        sample_num_Hneg = torch.ones_like(num_Hneg_) * 2
        # print('[sample points]:{}'.format(available_num))  # print log so that we can debug it.....
    else:
        sample_num_Epos = (percent * num_Epos_[keeps]).ceil()  # percent is the sigma in our paper
        sample_num_Hpos = (percent * num_Hpos_[keeps]).ceil()
        sample_num_Eneg = (percent * num_Eneg_[keeps]).ceil()
        sample_num_Hneg = (percent * num_Hneg_[keeps]).ceil()
        # print("sample_num_Hneg",sample_num_Hneg.shape)#5,1
        # print("sample_num_Hneg", sample_num_Hneg)#96

    #######################################################
    # ----------------- sample points ---------------------
    #######################################################
    # print("easy_positive_sets",easy_positive_sets.shape)#8 1 256 256
    # print("query_neg_sets",query_neg_sets.shape)# 8 1 256 256
    # print("query_neg_sets[keeps]",query_neg_sets[keeps].shape)#5 1 256 256
    # print("query_neg_sets[keeps]",query_neg_sets[keeps].shape)#5 1 256 256
    empty_dict = {}
    easy_positive_sets_N, flag0 = get_pixel_sets_N(easy_positive_sets[keeps], sample_num_Epos)
    if flag0 == False:
        return empty_dict, False
    easy_negative_sets_N, flag1 = get_pixel_sets_N(easy_negative_sets[keeps], sample_num_Eneg)
    if flag1 == False:
        return empty_dict, False
    hard_positive_sets_N, flag2 = get_pixel_sets_N(hard_positive_sets[keeps], sample_num_Hpos)
    if flag2 == False:
        return empty_dict, False
    hard_negative_sets_N, flag3 = get_pixel_sets_N(hard_negative_sets[keeps], sample_num_Hneg)
    if flag3 == False:
        return empty_dict, False

    # print("easy_positive_sets_N", easy_positive_sets_N.shape)
    # print("easy_negative_sets_N", easy_negative_sets_N.shape)
    # print("hard_positive_sets_N", hard_positive_sets_N.shape)
    # print("hard_negative_sets_N", hard_negative_sets_N.shape)
    # Record points number
    num_per_type = dict()
    num_per_type['Epos_num_'] = sample_num_Epos
    num_per_type['Hpos_num_'] = sample_num_Hpos
    num_per_type['Eneg_num_'] = sample_num_Eneg
    num_per_type['Hneg_num_'] = sample_num_Hneg
    # print("num_point_Epos", sample_num_Epos)
    # print("num_point_Hpos", sample_num_Hpos)
    # print("num_point_Eneg", sample_num_Eneg)
    # print("num_point_Hneg", sample_num_Hneg)

    #######################################################
    # ------------------- return data ---------------------
    #######################################################
    # query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
    # query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
    return_result = dict()
    return_result['keeps'] = keeps  # which proposal is preserved
    return_result['num_per_type'] = num_per_type
    return_result['query_pos_sets'] = query_pos_sets[keeps]  # query area for foreground
    return_result['query_neg_sets'] = query_neg_sets[keeps]  # query area for background
    return_result['easy_positive_sets_N'] = easy_positive_sets_N.to(dtype=torch.bool)
    return_result['easy_negative_sets_N'] = easy_negative_sets_N.to(dtype=torch.bool)
    return_result['hard_positive_sets_N'] = hard_positive_sets_N.to(dtype=torch.bool)
    return_result['hard_negative_sets_N'] = hard_negative_sets_N.to(dtype=torch.bool)
    return return_result, True


def write_tensormap(tensormap, file_name):
    map = tensormap.squeeze(0)
    map = map.squeeze(0)
    map_array = np.array(map)
    if np.max(map_array) == 255:
        map_Image = Image.fromarray((map_array).astype('uint8')).convert('L')
    else:
        map_Image = Image.fromarray((map_array * 255).astype('uint8')).convert('L')
    save_path = "/mnt/nas/sty/codes/Unsupervised/" + file_name
    map_Image.save(save_path)


def get_query_keys_myself(
        edges,
        masks=None,
        thred_u=0.1,
        scale_u=1.0,
        percent=0.3,
        fake=True):
    """
        Input
            edges: Tensor, cuda, Nx28x28
            masks: Tensor, cuda, Nx28x28 is the mask or prediction 0-1 0r [0,1]
    """
    #######################################################
    # ---------- some pre-processing -----------------------
    #######################################################
    masks = masks.cpu()  # to cpu, Nx28x28
    #######################################################
    # ---------- get query mask for each proposal ----------#
    #######################################################
    if fake:
        # write_tensormap(masks, "mask.png")
        query_pos_sets = masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = torch.logical_not(query_pos_sets)  # the background
        edges = edges.cpu()  # to cpu, Nx28x28 # 8 1 256 256
    else:
        pos_masks = torch.where(masks > (1 - thred_u), 1.0, 0.0).to(dtype=torch.bool)  # greater 0.9
        neg_mask = torch.where(masks < thred_u, 1.0, 0.0).to(dtype=torch.bool)  # less than 0.1
        query_pos_sets = pos_masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = neg_mask.to(dtype=torch.bool)  # 8 1 256 256
    #######################################################
    # ----------- get different types of keys -------------
    #######################################################
    # different sets, you can refer to the figure in https://blog.huiserwang.site/2022-03/Project-ContrastMask/ to easily understand.
    if fake:  # fakedata  with mask
        # for all region
        label_positive_sets = torch.where(masks > (1.0 - thred_u * scale_u), 1.0,
                                           0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        label_negative_sets = torch.where(masks < (thred_u * scale_u), 1.0, 0.0)
        easy_positive_sets = label_positive_sets
        easy_negative_sets = label_negative_sets
        hard_positive_sets = label_positive_sets
        hard_negative_sets = label_negative_sets

    else:
        # for novel(unseen), get keys according to cam, hard and easy are both sampled in the same sets, replace original sets
        unseen_positive_sets = torch.where(masks > (1.0 - thred_u * scale_u), 1.0,
                                           0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        unseen_negative_sets = torch.where(masks < (thred_u * scale_u), 1.0, 0.0)
        easy_positive_sets = unseen_positive_sets
        easy_negative_sets = unseen_negative_sets
        hard_positive_sets = unseen_positive_sets
        hard_negative_sets = unseen_negative_sets
    #######################################################
    # --------- determine the number of sampling ----------
    #######################################################
    # how many points can be sampled for all proposals for each type of sets
    num_Epos_ = easy_positive_sets.sum(dim=[2, 3])  # H, W to count points numbers E=easy, H=hard
    num_Hpos_ = hard_positive_sets.sum(dim=[2, 3])
    num_Eneg_ = easy_negative_sets.sum(dim=[2, 3])
    num_Hneg_ = hard_negative_sets.sum(dim=[2, 3])
    # if available points are less then 5 for each type, this proposal will be dropped out.
    available_num = torch.cat([num_Epos_, num_Eneg_, num_Hpos_, num_Hneg_])
    abandon_inds = torch.where(available_num < 5, 1, 0).reshape(4, -1)
    keeps = torch.logical_not(abandon_inds.sum(0).to(dtype=torch.bool))
    if True not in keeps:  # all proposals do not have enough points that can be sample. This is a extreme situation.
        # set the points number of all types sets to 2
        # sometimes, there would still raise an error. I will fix it later.
        sample_num_Epos = torch.ones_like(num_Epos_) * 2
        sample_num_Hpos = torch.ones_like(num_Hpos_) * 2
        sample_num_Eneg = torch.ones_like(num_Eneg_) * 2
        sample_num_Hneg = torch.ones_like(num_Hneg_) * 2.
    else:
        sample_num_Epos = (percent * num_Epos_[keeps]).ceil()  # percent is the sigma in our paper
        sample_num_Hpos = (percent * num_Hpos_[keeps]).ceil()
        sample_num_Eneg = (percent * num_Eneg_[keeps]).ceil()
        sample_num_Hneg = (percent * num_Hneg_[keeps]).ceil()

    #######################################################
    # ----------------- sample points ---------------------
    #######################################################
    empty_dict = {}
    easy_positive_sets_N, flag0 = get_pixel_sets_N_myself(easy_positive_sets[keeps], sample_num_Epos)
    if not flag0:
        return empty_dict, False
    easy_negative_sets_N, flag1 = get_pixel_sets_N_myself(easy_negative_sets[keeps], sample_num_Eneg)
    if not flag1:
        return empty_dict, False
    hard_positive_sets_N, flag2 = get_pixel_sets_N_myself(hard_positive_sets[keeps], sample_num_Hpos)
    if not flag2:
        return empty_dict, False
    hard_negative_sets_N, flag3 = get_pixel_sets_N_myself(hard_negative_sets[keeps], sample_num_Hneg)
    if not flag3:
        return empty_dict, False

    # Record points number
    num_per_type = dict()
    num_per_type['Epos_num_'] = sample_num_Epos
    num_per_type['Hpos_num_'] = sample_num_Hpos
    num_per_type['Eneg_num_'] = sample_num_Eneg
    num_per_type['Hneg_num_'] = sample_num_Hneg

    #######################################################
    # ------------------- return data ---------------------
    #######################################################
    return_result = dict()
    return_result['keeps'] = keeps  # which proposal is preserved
    return_result['num_per_type'] = num_per_type
    return_result['query_pos_sets'] = query_pos_sets[keeps]  # query area for foreground
    return_result['query_neg_sets'] = query_neg_sets[keeps]  # query area for background
    return_result['easy_positive_sets_N'] = easy_positive_sets_N.to(dtype=torch.bool)
    return_result['easy_negative_sets_N'] = easy_negative_sets_N.to(dtype=torch.bool)
    return_result['hard_positive_sets_N'] = hard_positive_sets_N.to(dtype=torch.bool)
    return_result['hard_negative_sets_N'] = hard_negative_sets_N.to(dtype=torch.bool)
    return return_result, True


def get_query_keys_noedge(
        masks=None,
        thred_u=0.1,
        scale_u=1.0,
        percent=0.3,
        fake=True):
    """
        Input
            edges: Tensor, cuda, Nx28x28
            masks: Tensor, cuda, Nx28x28 is the mask or prediction 0-1 0r [0,1]
    """
    #######################################################
    # ---------- some pre-processing -----------------------
    #######################################################
    masks = masks.detach().cpu()  # to cpu, Nx28x28 For 1206
    #######################################################
    # ---------- get query mask for each proposal ----------#
    #######################################################
    if fake:
        # write_tensormap(masks, "mask.png")
        query_pos_sets = masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = torch.logical_not(query_pos_sets)  # the background
        # write_tensormap(query_neg_sets, "query_neg.png")
        # write_tensormap(query_pos_sets, "query_pos.png")
    else:
        pos_masks = torch.where(masks > (1 - thred_u), 1.0, 0.0).to(dtype=torch.bool)  # greater 0.9
        neg_mask = torch.where(masks < thred_u, 1.0, 0.0).to(dtype=torch.bool)  # less than 0.1
        query_pos_sets = pos_masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = neg_mask.to(dtype=torch.bool)  # 8 1 256 256
    #######################################################
    # ----------- get different types of keys -------------
    #######################################################
    # different sets, you can refer to the figure in https://blog.huiserwang.site/2022-03/Project-ContrastMask/ to easily understand.
    if fake:  # fakedata  with mask
        positive_sets = torch.where(masks > 0, 1.0,
                                    0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        negative_sets = torch.where(masks <= 0, 1.0, 0.0)
        key_positive_sets = positive_sets
        key_negative_sets = negative_sets
        # print("key_positive_sets", key_positive_sets.shape) N C H W
    else:
        # for novel(unseen), get keys according to cam, hard and easy are both sampled in the same sets, replace original sets
        unseen_positive_sets = torch.where(masks > (1.0 - thred_u * scale_u), 1.0,
                                           0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        unseen_negative_sets = torch.where(masks < (thred_u * scale_u), 1.0, 0.0)
        key_positive_sets = unseen_positive_sets
        key_negative_sets = unseen_negative_sets
        # print("key_positive_sets",key_positive_sets.shape)
    #######################################################
    # --------- determine the number of sampling ----------
    #######################################################
    # how many points can be sampled for all proposals for each type of sets
    num_pos_ = key_positive_sets.sum(dim=[2, 3])
    num_neg_ = key_negative_sets.sum(dim=[2, 3])
    # print("num_Epos_",num_Epos_)#
    # print("num_Hpos_",num_Hpos_)
    # print("num_Eneg_",num_Eneg_)
    # print("num_Hneg_",num_Hneg_)
    # print("num_Hneg_",num_Hneg_.shape)#8 1
    # if available points are less then 5 for each type, this proposal will be dropped out.
    available_num = torch.cat([num_pos_, num_neg_])
    # print("available_num",available_num.shape)#32 , 1 16,1
    # print("available_num", available_num)#4 , 8
    abandon_inds = torch.where(available_num < 5, 1, 0).reshape(2, -1)
    # print("abandon_inds", abandon_inds.shape)#4 , 8
    # print("abandon_inds", abandon_inds)#4 , 8
    keeps = torch.logical_not(abandon_inds.sum(0).to(dtype=torch.bool))
    # print("keeps",keeps.shape)#8
    # print("keeps_show",keeps)#[T,F] to find which case hase True
    if True not in keeps:  # all proposals do not have enough points that can be sample. This is a extreme situation.
        # set the points number of all types sets to 2
        # sometimes, there would still raise an error. I will fix it later.
        sample_num_pos = torch.ones_like(num_pos_) * 2
        sample_num_neg = torch.ones_like(num_neg_) * 2
        # print('[sample points]:{}'.format(available_num))  # print log so that we can debug it.....
    else:
        # print("num_pos_[keeps]",num_pos_.shape) #[8,1]
        sample_num_pos = (percent * num_pos_[keeps]).ceil()
        sample_num_neg = (0.1 * percent * num_neg_[keeps]).ceil()
        # print("sample_num_Hneg",sample_num_Hneg.shape)#5,1
        # print("sample_num_Epos", sample_num_Epos)#96
        # print("sample_num_Hpos", sample_num_Hpos)#96
        # print("sample_num_Eneg", sample_num_Eneg)#96
        # print("sample_num_Hneg", sample_num_Hneg)#96
        # need to get a better balance points

    #######################################################
    # ----------------- sample points ---------------------
    #######################################################
    # print("easy_positive_sets",easy_positive_sets.shape)#8 1 256 256
    # print("query_neg_sets",query_neg_sets.shape)# 8 1 256 256
    # print("query_neg_sets[keeps]",query_neg_sets[keeps].shape)#5 1 256 256
    # print("query_neg_sets[keeps]",query_neg_sets[keeps].shape)#5 1 256 256
    empty_dict = {}
    easy_positive_sets_N, flag0 = get_pixel_sets_N_myself(key_positive_sets[keeps], sample_num_pos)
    # print("easy_positive_sets_Nshape",easy_positive_sets_N.shape)
    # write_tensormap(easy_positive_sets_N, "easy_positive_sets_N.png")
    if flag0 == False:
        return empty_dict, False

    easy_negative_sets_N, flag1 = get_pixel_sets_N_myself(key_negative_sets[keeps], sample_num_neg)
    # write_tensormap(easy_negative_sets_N, "easy_negative_sets_N.png")
    if flag1 == False:
        return empty_dict, False

    num_per_type = dict()
    num_per_type['pos_num_'] = sample_num_pos
    num_per_type['neg_num_'] = sample_num_neg
    #######################################################
    # ------------------- return data ---------------------
    #######################################################
    # query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
    # query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])

    return_result = dict()
    return_result['keeps'] = keeps  # which proposal is preserved
    return_result['num_per_type'] = num_per_type
    # return_result['query_pos_sets'] = query_pos_sets[keeps]  # query area for foreground
    # return_result['query_neg_sets'] = query_neg_sets[keeps]  # query area for background
    return_result['easy_positive_sets_N'] = easy_positive_sets_N.to(dtype=torch.bool)
    return_result['easy_negative_sets_N'] = easy_negative_sets_N.to(dtype=torch.bool)
    return return_result, True


def get_pixel_sets_N(src_sets, select_num):
    return_ = []
    if isinstance(src_sets, torch.Tensor):
        bs, c, h, w = src_sets.shape
        # print("src_sets_shape",src_sets.shape)#5 1 256 256
        # print("select_num", select_num.shape)  # 5 1
        # print("torch.where(src_sets > 0.5, 1, 0)",torch.where(src_sets > 0.5, 1, 0).shape)
        flag = True
        if torch.where(src_sets > 0.5, 1, 0).shape[0] == 0:
            flag = False
            return src_sets, False
        keeps_all = torch.where(src_sets > 0.5, 1, 0).reshape(bs, -1)
        # print("keeps_all",keeps_all.shape)#5,65536
        for idx, keeps in enumerate(keeps_all):
            keeps_init = np.zeros_like(keeps)
            # print("gets_set_N",keeps_init.shape)#65536
            src_set_index = np.arange(len(keeps))
            # print("src_set_index",src_set_index.shape)#65536
            src_set_index_keeps = src_set_index[keeps.numpy().astype(np.bool)]
            # print("src_set_index_keeps",src_set_index_keeps.shape)# (1371,) 5 points for each image
            resultList = random.sample(range(0, len(src_set_index_keeps)), int(select_num[idx]))
            # print("len(resultList)",len(resultList))#412
            src_set_index_keeps_select = src_set_index_keeps[resultList]
            # print("src_set_index_keeps_select",src_set_index_keeps_select.shape)#412,
            keeps_init[src_set_index_keeps_select] = 1
            # print("keeps_init",keeps_init.shape)#65536 256*256
            return_.append(keeps_init.reshape(1, h, w))
    else:
        raise ValueError(f'only tensor is supported!')
    # print("Len_return_",len(return_))
    # print("Len_return_",return_[0].shape)
    return torch.tensor(return_) * src_sets, flag


def get_pixel_sets_N_myself(src_sets, select_num):
    return_ = []
    if isinstance(src_sets, torch.Tensor):
        bs, c, h, w = src_sets.shape
        flag = True
        if torch.where(src_sets > 0.5, 1, 0).shape[0] == 0:
            flag = False
            return src_sets, False
        keeps_all = torch.where(src_sets > 0.5, 1, 0).reshape(bs, -1)  # get the right area point
        for idx, keeps in enumerate(keeps_all):
            keeps_init = np.zeros_like(keeps.cpu())  # For 1204
            src_set_index = np.arange(len(keeps))
            src_set_index_keeps = src_set_index[keeps.cpu().numpy().astype(np.bool)]  # For 1204
            select_num[idx] = int(select_num[idx]) if int(select_num[idx]) < 500 else 500
            resultList = random.sample(range(0, len(src_set_index_keeps)), int(select_num[idx]))
            src_set_index_keeps_select = src_set_index_keeps[resultList]
            keeps_init[src_set_index_keeps_select] = 1
            return_.append(torch.tensor(keeps_init).reshape(1, h, w))
    else:
        raise ValueError(f'only tensor is supported!')
    return_ = [aa.tolist() for aa in return_]  # For 1207
    return torch.tensor(return_) * src_sets, flag


def get_pixel_sets_distrans(src_sets, radius=2):
    """
        src_sets: shape->[N, 28, 28]
    """
    if isinstance(src_sets, torch.Tensor):
        src_sets = src_sets.numpy()
    if isinstance(src_sets, np.ndarray):
        keeps = []
        for src_set in src_sets:
            keep = distance_transform_edt(np.logical_not(src_set))
            keep = keep < radius
            keeps.append(keep.astype(np.float))
    else:
        raise ValueError(f'only np.ndarray is supported!')
    return torch.tensor(keeps).to(dtype=torch.long)


def normalize_zero_to_one(imgs):
    if isinstance(imgs, torch.Tensor):
        bs, h, w = imgs.shape
        imgs_mins = getattr(imgs.view(bs, -1).min(1), 'values').view(bs, 1, 1)
        imgs_maxs = getattr(imgs.view(bs, -1).max(1), 'values').view(bs, 1, 1)
        return (imgs - imgs_mins) / (imgs_maxs - imgs_mins)
    else:
        raise TypeError(f'Only tensor is supported!')


def mask2edge(seg):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=seg.device).reshape(1, 1, 3, 3).requires_grad_(False)
    edge_targets = F.conv2d(seg, laplacian_kernel, padding=1)
    edge_targets = edge_targets.clamp(min=0)
    edge_targets[edge_targets > 0.1] = 1
    edge_targets[edge_targets <= 0.1] = 0
    return edge_targets


if __name__ == '__main__':
    mask = np.array(Image.open("/mnt/nas/sty/codes/Unsupervised/111.png").convert('L'))
    print("mask_shape", mask.shape)
    mask_tensor = np.expand_dims(mask, 0)
    mask_tensor = np.expand_dims(mask_tensor, 0)
    print("mask_tensor", mask_tensor.shape)
    mask_torch = torch.tensor(mask_tensor)
    write_tensormap(mask_torch, "mask_orgin.png")
    edge_targets = mask2edge(mask_torch * 1.0)
    print("edge_targets_unique", torch.unique(edge_targets))
    print("edageshape", edge_targets.shape)
    edge = edge_targets.squeeze(0)
    edge = edge.squeeze(0)
    print("edge_shape", edge.shape)

    edge_array = np.array(edge)
    edge_Image = Image.fromarray((edge_array * 255).astype('uint8')).convert('L')
    edge_Image.save("/mnt/nas/sty/codes/Unsupervised/111_edage.png")
    return_result, flag = get_query_keys_myself(edge_targets, masks=mask_torch / 255, thred_u=0.1, scale_u=1.0,
                                                percent=0.3, fake=True)
