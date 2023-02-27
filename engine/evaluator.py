import os
import cv2
import numpy as np
import time
from tqdm import tqdm

import torch
import multiprocessing as mp

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.img_utils import pad_image_to_shape, normalize

logger = get_logger()


class Evaluator(object):
    def __init__(self, dataset, class_num, image_mean, image_std, network,
                 multi_scales, is_flip, devices,
                 verbose=False, save_path=None, show_image=False):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.image_mean = image_mean
        self.image_std = image_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image

    def run(self, model_path, model_indice, log_file, log_file_link):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            if "epoch-last.pth" in models:
                models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)


            for idx, m in enumerate(models):
                print(m)
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'epoch-%s.pth' % model_indice)):
                models = [os.path.join(model_path,
                                       'epoch-%s.pth' % model_indice), ]
            else:
                models = [None]

        results = open(log_file, 'a')
        link_file(log_file, log_file_link)

        for model in models:
            logger.info("Load Model: %s" % model)
            if model is not None:
                self.val_func = load_model(self.network, model)
            else:
                logger.info("No model is loaded !!!!!!!")
                self.val_func = self.network
            #from IPython import embed; embed()
            if len(self.devices ) == 1:
                result_line = self.single_process_evalutation()
            else:
                result_line = self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()

    def single_process_evalutation(self):
        start_eval_time = time.perf_counter()

        logger.info(
                'GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []
        for idx in tqdm(range(self.ndata)):
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd,self.devices[0])
            all_results.append(results_dict)
        result_line = self.compute_metric(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line



    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):

            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info(
                'GPU %s handle %d data.' % (device, len(shred_list)))

            p = self.context.Process(target=self.worker,
                                     args=(shred_list, device))
            procs.append(p)

        for p in procs:

            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line = self.compute_metric(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line

    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (
            device, time.time() - start_load_time))

        for idx in shred_list:
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd, device)
            self.results_queue.put(results_dict)

    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError

    # evaluate the whole image at once
    def whole_eval(self, img, output_size, device=None):
        processed_pred = np.zeros(
            (output_size[0], output_size[1], self.class_num))

        for s in self.multi_scales:
            scaled_img = cv2.resize(img, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_LINEAR)
            scaled_img = self.process_image(scaled_img, None)
            pred = self.val_func_process(scaled_img, device)
            pred = pred.permute(1, 2, 0)
            processed_pred += cv2.resize(pred.cpu().numpy(),
                                         (output_size[1], output_size[0]),
                                         interpolation=cv2.INTER_LINEAR)

        pred = processed_pred.argmax(2)

        return pred

    # slide the window to evaluate the image
    def sliding_eval(self, img, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process(img_scale,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process(self, img, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        if long_size <= min(crop_size[0], crop_size[1]):
            input_data, margin = self.process_image(img, crop_size)     # pad image
            score = self.val_func_process(input_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride_0 = int(np.ceil(crop_size[0] * stride_rate))
            stride_1 = int(np.ceil(crop_size[1] * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride_0)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride_1)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride_1
                    s_y = grid_yidx * stride_0
                    e_x = min(s_x + crop_size[1], pad_cols)
                    e_y = min(s_y + crop_size[0], pad_rows)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data)
                score = score[0]

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    score_flip = self.val_func(input_data)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                # score = torch.exp(score)
                # score = score.data

        return score

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img


    # add new funtion for rgb and disparity segmentation
    def sliding_eval_rgbdsip(self, img, disp, camera_params, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            disp_scale = cv2.resize(disp, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_NEAREST)
            camera_params['scale'] = s
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process_rgbdisp(img_scale, disp_scale, camera_params,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process_rgbdisp(self, img, disp, camera_params,ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, input_disp, margin = self.process_image_rgbd(img, disp, crop_size)
            score = self.val_func_process_rgbd(input_data, input_disp, camera_params, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)
            disp_pad, margin = pad_image_to_shape(disp, crop_size,
                                                  cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    disp_sub = disp_pad[s_y:e_y, s_x: e_x]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, input_disp, tmargin = self.process_image_rgbd(img_sub, disp_sub,crop_size)
                    temp_score = self.val_func_process_rgbd(input_data, input_disp, camera_params, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output
    

    
    
    def val_func_process_rgbdisp(self, input_data, input_disp, camera_params, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
        
        
        input_disp = np.ascontiguousarray(input_disp[None, :, :, :],
                                          dtype=np.float32)
        input_disp = torch.FloatTensor(input_disp).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data, input_disp, camera_params)
                score = score[0]

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_disp = input_disp.flip(-1)
                    score_flip = self.val_func(input_data, input_disp, camera_params)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)
                # score = score.data

        return score

    # add new funtion for rgb and depth segmentation
    def sliding_eval_rgbdepth(self, img, disp, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))


        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            if len(disp.shape) == 2:
                disp_scale = cv2.resize(disp, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_NEAREST)
            else :
                disp_scale = cv2.resize(disp, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process_rgbdepth(img_scale, disp_scale,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process_rgbdepth(self, img, disp, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, input_disp, margin = self.process_image_rgbd(img, disp, crop_size)
            score = self.val_func_process_rgbdepth(input_data, input_disp, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)
            disp_pad, margin = pad_image_to_shape(disp, crop_size,
                                                  cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    if len(disp_pad.shape) == 2:
                        disp_sub = disp_pad[s_y:e_y, s_x: e_x]
                    else:
                        disp_sub = disp_pad[s_y:e_y, s_x: e_x,:]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, input_disp, tmargin = self.process_image_rgbd(img_sub, disp_sub,crop_size)
                    temp_score = self.val_func_process_rgbdepth(input_data, input_disp, device)

                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process_rgbdepth(self, input_data, input_disp,  device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
    
        input_disp = np.ascontiguousarray(input_disp[None, :, :, :],
                                          dtype=np.float32)
        input_disp = torch.FloatTensor(input_disp).cuda(device)
    
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data, input_disp)
                score = score[0]
            
                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_disp = input_disp.flip(-1)
                    score_flip = self.val_func(input_data, input_disp)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)
                # score = score.data
    
        return score
    # for rgbd segmentation
    def process_image_rgbd(self, img, disp, crop_size=None):
        p_img = img
        p_disp = disp
    
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)
    
        p_img = normalize(p_img, self.image_mean, self.image_std)
        if len(disp.shape) == 2:
            p_disp = normalize(p_disp, 0, 1)
        else:
            p_disp  = normalize(p_disp, self.image_mean, self.image_std)
            
    
        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
        
            p_disp, _ = pad_image_to_shape(p_disp, crop_size,
                                                cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            if len(disp.shape)  ==  2:
                p_disp = p_disp[np.newaxis, ...]
            else:
                p_disp = p_disp.transpose(2, 0, 1)
        
            return p_img, p_disp, margin
    
        p_img = p_img.transpose(2, 0, 1)

        if len(disp.shape) == 2:
            p_disp = p_disp[np.newaxis, ...]
        else:
            p_disp = p_disp.transpose(2, 0, 1)
    
        return p_img, p_disp
