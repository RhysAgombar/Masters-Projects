import numpy as np
import scipy.io as scio
import os
import argparse
import pickle
from sklearn import metrics
import json
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import argparse

DATA_DIR = '../Data'
NORMALIZE = True
DECIDABLE_IDX = 4 ## First 4 frames are ignored
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_grad_loss(gen_frame, gt_frame, alpha):

    gr = gen_frame[0].unsqueeze(0).unsqueeze(0)
    gg = gen_frame[1].unsqueeze(0).unsqueeze(0)
    gb = gen_frame[2].unsqueeze(0).unsqueeze(0)

    tr = gt_frame.squeeze(0)[0].unsqueeze(0).unsqueeze(0)
    tg = gt_frame.squeeze(0)[1].unsqueeze(0).unsqueeze(0)
    tb = gt_frame.squeeze(0)[2].unsqueeze(0).unsqueeze(0)

    # Filter according to paper equations
    x_filter = torch.tensor([[0.0,0.0,0.0],[-1.0,1.0,0.0],[0.0,0.0,0.0]]).unsqueeze(0).unsqueeze(0).to(device) # Using 3x3 filters with the same data because 2x2 gives errors
    y_filter = torch.tensor([[0.0,-1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).unsqueeze(0).unsqueeze(0).to(device) # Should result in the same thing

    gr_x = torch.abs(F.conv2d(gr, x_filter, padding=1))
    gg_x = torch.abs(F.conv2d(gg, x_filter, padding=1))
    gb_x = torch.abs(F.conv2d(gb, x_filter, padding=1))

    gr_y = torch.abs(F.conv2d(gr, y_filter, padding=1))
    gg_y = torch.abs(F.conv2d(gg, y_filter, padding=1))
    gb_y = torch.abs(F.conv2d(gb, y_filter, padding=1))

    tr_x = torch.abs(F.conv2d(tr, x_filter, padding=1))
    tg_x = torch.abs(F.conv2d(tg, x_filter, padding=1))
    tb_x = torch.abs(F.conv2d(tb, x_filter, padding=1))

    tr_y = torch.abs(F.conv2d(tr, y_filter, padding=1))
    tg_y = torch.abs(F.conv2d(tg, y_filter, padding=1))
    tb_y = torch.abs(F.conv2d(tb, y_filter, padding=1))

    gen_x = torch.cat((gr_x,gg_x,gb_x), dim=1)
    gen_x = torch.abs((gen_x - gen_x.min())/(gen_x.max() - gen_x.min()))

    gen_y = torch.cat((gr_y,gg_y,gb_y), dim=1)
    gen_y = torch.abs((gen_y - gen_y.min())/(gen_y.max() - gen_y.min()))

    gt_x = torch.cat((tr_x,tg_x,tb_x), dim=1)
    gt_x = torch.abs((gt_x - gt_x.min())/(gt_x.max() - gt_x.min()))

    gt_y = torch.cat((tr_y,tg_y,tb_y), dim=1)
    gt_y = torch.abs((gt_y - gt_y.min())/(gt_y.max() - gt_y.min()))

    gen_der = gen_x + gen_y
    gt_der = gt_x + gt_y

    out_im = torch.cat((gen_der.detach().cpu().squeeze(0), gt_der.detach().cpu().squeeze(0)), dim=2) 

    loss = torch.mean(torch.abs(gen_x - gt_x) + torch.abs(gen_y - gt_y))

    return loss, out_im

## Taken from paper's implementation
class GroundTruthLoader(object):
    AVENUE = 'avenue'
    PED1 = 'ped1'
    PED1_PIXEL_SUBSET = 'ped1_pixel_subset'
    PED2 = 'ped2'
    ENTRANCE = 'enter'
    EXIT = 'exit'
    SHANGHAITECH = 'shanghaitech'
    SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech/testing/test_frame_mask')
    TOY_DATA = 'toydata'
    TOY_DATA_LABEL_PATH = os.path.join(DATA_DIR, TOY_DATA, 'toydata.json')

    NAME_MAT_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/avenue.mat'),
        PED1: os.path.join(DATA_DIR, 'ped1/ped1.mat'),
        PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/enter.mat'),
        EXIT: os.path.join(DATA_DIR, 'exit/exit.mat')
    }

    NAME_FRAMES_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/testing/frames'),
        PED1: os.path.join(DATA_DIR, 'ped1/testing/frames'),
        PED2: os.path.join(DATA_DIR, 'ped2/testing/frames'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/testing/frames'),
        EXIT: os.path.join(DATA_DIR, 'exit/testing/frames')
    }

    def __init__(self, mapping_json=None):
        """
        Initial a ground truth loader, which loads the ground truth with given dataset name.

        :param mapping_json: the mapping from dataset name to the path of ground truth.
        """

        if mapping_json is not None:
            with open(mapping_json, 'rb') as json_file:
                self.mapping = json.load(json_file)
        else:
            self.mapping = GroundTruthLoader.NAME_MAT_MAPPING

    def __call__(self, dataset, remove_ratio):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        elif dataset == GroundTruthLoader.TOY_DATA:
            gt = self.__load_toydata_gt()
        else:
            gt = self.__load_ucsd_avenue_subway_gt(dataset, remove_ratio)
        return gt

    def __load_ucsd_avenue_subway_gt(self, dataset, remove_ratio):
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            # video_name = video_name_template.format(sub_video_number)
            video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

            length = len(os.listdir(video_name))

            return length

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):
            length = get_video_length(i) * remove_ratio

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            ### Modification for number of skipped frames ###
            holder = sub_video_gt[0::remove_ratio] ## get every fourth frame
            sub_video_gt = holder

            gt.append(sub_video_gt)

        return gt

    @staticmethod
    def __load_shanghaitech_gt():
        video_path_list = os.listdir(GroundTruthLoader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video))
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video)))

        return gt

    @staticmethod
    def __load_toydata_gt():
        with open(GroundTruthLoader.TOY_DATA_LABEL_PATH, 'r') as gt_file:
            gt_dict = json.load(gt_file)

        gt = []
        for video, video_info in gt_dict.items():
            length = video_info['length']
            video_gt = np.zeros((length,), dtype=np.uint8)
            sub_gt = np.array(np.matrix(video_info['gt']))

            for anomaly in sub_gt:
                start = anomaly[0]
                end = anomaly[1] + 1
                video_gt[start: end] = 1
            gt.append(video_gt)
        return gt

    @staticmethod
    def get_pixel_masks_file_list(dataset):
        # pixel mask folder
        pixel_mask_folder = os.path.join(DATA_DIR, dataset, 'pixel_masks')
        pixel_mask_file_list = os.listdir(pixel_mask_folder)
        pixel_mask_file_list.sort()

        # get all testing videos
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        # get all testing video names with pixel masks
        pixel_video_ids = []
        ids = 0
        for pixel_mask_name in pixel_mask_file_list:
            while ids < len(video_list):
                if video_list[ids] + '.npy' == pixel_mask_name:
                    pixel_video_ids.append(ids)
                    ids += 1
                    break
                else:
                    ids += 1

        assert len(pixel_video_ids) == len(pixel_mask_file_list)

        for i in range(len(pixel_mask_file_list)):
            pixel_mask_file_list[i] = os.path.join(pixel_mask_folder, pixel_mask_file_list[i])

        return pixel_mask_file_list, pixel_video_ids

## Taken from paper's implementation
class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None, loss_file=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset
        self.loss_file = loss_file

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, loss file = {}, auc = {}'.format(self.dataset, self.loss_file, self.auc)

def psnr(gen_frames, gt_frames):
    shape = gen_frames.shape
    num_pixels = torch.tensor([shape[1] * shape[2] * shape[3]], dtype=torch.float32)
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = (gt_frames - gen_frames).pow(2)

    batch_errors = 10 * torch.log10(1 / ((1 / num_pixels) * torch.sum(square_diff, [1, 2, 3])))
    return torch.mean(batch_errors)

def psnr_np(gen_frames, gt_frames):
    shape = gen_frames.shape
    num_pixels = shape[1] * shape[2] * shape[3]
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = np.power((gt_frames - gen_frames),2)

    one = (1 / num_pixels)
    t1 = np.sum(square_diff, axis=1)
    t2 = np.sum(t1, axis=1)
    t3 = np.sum(t2, axis=1)
    two = t3


    batch_errors = 10 * np.log10(1 / (one * two))
    return np.mean(batch_errors)

def psnr_list(gen_frames, gt_frames):
    psnrs_list = []

    for j in range(len(gt_frames)):
        length = len(gt_frames[j])
        psnrs = np.empty(shape=(length,), dtype=np.float32)
        for i in range(DECIDABLE_IDX, length):
            ps = psnr_np(gen_frames[j][i], gt_frames[j][i])
            psnrs[i] = ps
        
        psnrs[0:DECIDABLE_IDX] = psnrs[DECIDABLE_IDX]
        psnrs_list += [psnrs]

    return psnrs_list

## AUC Calculations
def compute_auc(psnr_records, gt):

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(2,3): #num_videos):
        distance = psnr_records[i]

        if NORMALIZE:
            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            # distance = 1 - distance

        scores = distance[DECIDABLE_IDX:] #np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
        labels = gt[i][DECIDABLE_IDX:] #np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    return auc 
    #results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)