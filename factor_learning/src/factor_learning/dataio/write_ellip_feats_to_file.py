# Copyright (c) Facebook, Inc. and its affiliates.

from factor_learning.dataio.DigitImageTfDataset import DigitImageTfDataset
import matplotlib.pyplot as plt
import imageio
import cv2
from PIL import Image
import os
import json
import math
import numpy as np
import hydra

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from datetime import datetime

def feat_from_ellip_params(img_orig, ellip_center, ellip_size, ellip_ang_rad):

    im_rows, im_cols, im_ch = img_orig.shape

    feat = torch.FloatTensor(6, 1)
    feat[0] = torch.FloatTensor([ellip_center[0]/im_cols])
    feat[1] = torch.FloatTensor([ellip_center[1]/im_rows])
    feat[2] = torch.FloatTensor([ellip_size[0]/im_cols])
    feat[3] = torch.FloatTensor([ellip_size[1]/im_rows])
    feat[4] = torch.cos(torch.FloatTensor([ellip_ang_rad]))
    feat[5] = torch.sin(torch.FloatTensor([ellip_ang_rad]))

    return feat


def get_ellip_params(ellipse):

    # center_offset: offset_x (cols), offset_y (rows)
    # size_scale: size_x, size_y

    center_offset = [0, 0]
    size_scale = [0.5, 0.5]

    ellip_center = (int(ellipse[0][0] + center_offset[0]),
                    int(ellipse[0][1] + center_offset[1]))
    ellip_size = (int(ellipse[1][0] * size_scale[0]),
                  int(ellipse[1][1] * size_scale[1]))
    ellip_ang = ellipse[2]

    return (ellip_center, ellip_size, ellip_ang)


def ellipse_contour_feat(img_orig, img_thresh, vis_ellipse=True, axes=None, labels=None):

    img_cnt = img_orig.copy()

    contours, hierarchy = cv2.findContours(
        img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort detected contours according to area
    cnt_area_list = []
    for (idx, cnt) in enumerate(contours):
        area = cv2.contourArea(cnt)
        cnt_area_list.append(area)
    sort_idxs = np.argsort(np.array(cnt_area_list))

    # fit ellipse to contour of max size
    cnt = contours[sort_idxs[-1]]
    ellipse = cv2.fitEllipse(cnt)

    ellip_center, ellip_size, ellip_ang = get_ellip_params(ellipse)
    ellip_ang_rad = ellip_ang * np.pi / 180

    if (vis_ellipse):
        cv2.ellipse(img_cnt, ellip_center, ellip_size,
                    ellip_ang, 0.0, 360.0, (0, 0, 0), 2)
        cv2.circle(img_cnt, ellip_center, 4, (0, 0, 255), -1)

        if axes is None:
            cv2.imshow("img_orig", img_orig)
            cv2.imshow("img_thresh", img_thresh)
            cv2.imshow("img_ellip", img_cnt)
            cv2.waitKey(20)
        else:
            axes[0].imshow(img_orig)
            axes[1].imshow(img_cnt)
            axes[0].set_title(labels[0])
            axes[1].set_title(labels[1])

    feat = feat_from_ellip_params(
        img_orig, ellip_center, ellip_size, ellip_ang_rad)

    return feat


def ellip_feat_extractor(img, mean_img, vis_feat_flag=False, axes=None, labels=None):

    img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mean_img = cv2.cvtColor(mean_img, cv2.COLOR_RGB2BGR)

    min_diff = 5
    img_diff = cv2.subtract(img, mean_img)
    img_diff[img_diff < min_diff] = 0
    
    img_diff_ch = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
    # img_diff_ch = (0.5*(img_diff[:, :, 1] + img_diff[:, :, 2])).astype(np.uint8)
    thresh = cv2.threshold(img_diff_ch, 0, 255, cv2.THRESH_OTSU)[1]

    feat = ellipse_contour_feat(img, thresh, vis_feat_flag, axes, labels)

    return feat

def write_feats_to_file(dataset, mean_img):
    
    num_data = len(dataset)
    for idx in range(0, num_data):

        img, pose = dataset[idx]
        feat = ellip_feat_extractor(img, mean_img, True)

        item_loc = dataset.get_item_loc(idx)
        dstfile = "{0}_feat_ellip.pt".format(item_loc)
        torch.save(feat.data, dstfile)

        print("Writing feature to: {0}".format(dstfile))

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
def main():
    global DATASET_NAME

    # DATASET_NAME = "20200619_disc_6in_new_firmware"
    # DATASET_NAME = "20200624_pushing-6in-disc-straight-line"
    # DATASET_NAME = "20200624_pushing-6in-disc-curves"
    # DATASET_NAME = "20200624_pushing-6in-disc-trial1"
    # DATASET_NAME = "20200928_rectangle-pushing-edges"
    # DATASET_NAME = "20200928_rectangle-pushing-corners"
    # DATASET_NAME = "20200928_ellipse-pushing-straight"
    DATASET_NAME = "20200928_ellipse-pushing"

    downsample_imgs = True

    # read mean img
    mean_img = imageio.imread("{0}/local/resources/digit/{1}/mean_img.png".format(BASE_PATH, DATASET_NAME))
    if downsample_imgs:
        mean_img = cv2.resize(mean_img, (64, 64))
    
    # load dataset
    srcdir_dataset = "{0}/local/datasets/{1}/".format(BASE_PATH, DATASET_NAME)
    subdirs = next(os.walk(srcdir_dataset))[1]

    for subdir in subdirs:
        transform = transforms.ToTensor()
        dataset = DigitImageTfDataset(
            "{0}/{1}".format(srcdir_dataset, subdir), transform=transform, downsample_imgs=downsample_imgs)
        write_feats_to_file(dataset, mean_img)

if __name__ == "__main__":

    main()
