# Copyright (c) Facebook, Inc. and its affiliates.

from factor_learning.utils import utils
from factor_learning.dataio.DigitImageTfDataset import DigitImageTfDataset
from factor_learning.dataio.DigitImageTfPairsDataset import DigitImageTfPairsDataset

from subprocess import call
import os
from scipy import linalg
import numpy as np
import cv2
from PIL import Image
import math

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

import seaborn as sns
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle


plt.rcParams.update({'font.size': 14})
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def visualize_correlation(feat_ij, pose_ij):
    data_tensor = torch.cat([feat_ij, pose_ij], 1)
    data = data_tensor.data.numpy()

    data_frame = pd.DataFrame(data)
    data_frame.columns = ['$f_{ij}[0]$: cx', '$f_{ij}[1]$: cy', '$f_{ij}[2]$: szx',
                          '$f_{ij}$[3]: szy', '$f_{ij}[4]$: c$\\alpha$', '$f_{ij}[5]$: s$\\alpha$',
                          '$T_{ij}[0]$: tx', '$T_{ij}[1]$: ty', '$T_{ij}[2]$: c$\\theta$', '$T_{ij}[3]$: s$\\theta$']

    corr_matrix = data_frame.corr()

    # plot correlation scatter plot
    fig1 = plt.figure()
    scatter_matrix(data_frame)
    plt.show()

    # plot correlation matrix
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    heatmap = sns.heatmap(corr_matrix,
                          square=True,
                          linewidths=.5,
                          cmap='coolwarm',
                          cbar_kws={'shrink': .4,
                                    'ticks': [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1,
                          annot=True,
                          annot_kws={'size': 12})

    # add the column names as labels
    ax.set_yticklabels(corr_matrix.columns, rotation=0)
    ax.set_xticklabels(corr_matrix.columns)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.show()


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


def tune_ellipse(ellipse):

    # center_offset: offset_x (cols), offset_y (rows)
    # size_scale: size_x, size_y
    center_offset = [-35, 10]
    size_scale = [0.6, 0.6]

    if (DATASET_NAME == "20200624_pushing-6in-disc-straight-line"):
        center_offset = [-30, 10]
    if (DATASET_NAME == "20200624_pushing-6in-disc-curves"):
        center_offset = [-25, 10]

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

    ellip_center, ellip_size, ellip_ang = tune_ellipse(ellipse)
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


def ellip_feat_extractor(img_batch, mean_img, vis_feat_flag=False, axes=None, labels=None):
    num_batch = img_batch.shape[0]
    feat_dim = 6
    feat_batch = torch.zeros(num_batch, feat_dim)

    for img_idx in range(num_batch):
        img = transforms.ToPILImage(mode='RGB')(img_batch[img_idx, :, :, :])
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = img.astype(np.uint8)

        min_diff = 5
        img_diff = cv2.subtract(img, mean_img)
        img_diff[img_diff < min_diff] = 0

        # img_diff_ch = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
        img_diff_ch = (
            0.5*(img_diff[:, :, 0] + img_diff[:, :, 1])).astype(np.uint8)
        thresh = cv2.threshold(img_diff_ch, 0, 255, cv2.THRESH_OTSU)[1]

        feat = ellipse_contour_feat(img, thresh, vis_feat_flag, axes, labels)
        feat_batch[img_idx, :] = feat.flatten()

    return feat_batch


def network_input_correlations(dataloader, mean_img):

    num_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)
    feats, poses = None, None
    for batch_idx, data in enumerate(dataloader):

        print("[network_input_correlations] batch {0} / {1} of size {2}".format(
            batch_idx, num_batches-1, dataloader.batch_size))

        img_i, img_j, pose_i, pose_j = data

        feat_i = ellip_feat_extractor(img_i, mean_img)
        feat_j = ellip_feat_extractor(img_j, mean_img)

        feat_ij = feat_j - feat_i
        pose_ij_xyh = utils.tf2d_between(pose_i, pose_j)
        pose_ij = utils.tf2d_net_input(pose_ij_xyh)

        feat_ji = feat_i - feat_j
        pose_ji_xyh = utils.tf2d_between(pose_j, pose_i)
        pose_ji = utils.tf2d_net_input(pose_ji_xyh)

        if (batch_idx == 0):
            feats = torch.cat([feat_ij, feat_ji], 0)
            poses = torch.cat([pose_ij, pose_ji], 0)
        else:
            feats = torch.cat([feats, feat_ij, feat_ji], 0)
            poses = torch.cat([poses, pose_ij, pose_ji], 0)

    visualize_correlation(feats, poses)


def write_feats_to_file(dataset, mean_img):

    num_data = len(dataset)
    for idx in range(0, num_data):

        img, pose = dataset[idx]
        img.unsqueeze_(0)
        feat = ellip_feat_extractor(img, mean_img, True)

        item_loc = dataset.get_item_loc(idx)
        dstfile = "{0}_feat_ellip.pt".format(item_loc)
        torch.save(feat.data, dstfile)

        print("Writing feature to: {0}".format(dstfile))


def visualize_img_feat_tf_pairs(dataloader, mean_img):
    num_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)
    feats, poses = None, None
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    for batch_idx, data in enumerate(dataloader):

        img_i, img_j, pose_i, pose_j = data
        pose_ij_xyh = utils.tf2d_between(pose_i, pose_j)
        pose_ij = utils.tf2d_net_input(pose_ij_xyh)

        nrows = 3
        ncols = 3
        gs = GridSpec(nrows, ncols, figure=fig)

        axes = [None] * (nrows*ncols)
        axes[0] = fig.add_subplot(gs[0, 0])
        axes[1] = fig.add_subplot(gs[0, 1])
        axes[2] = fig.add_subplot(gs[0, 2])
        axes[3] = fig.add_subplot(gs[1, 0])
        axes[4] = fig.add_subplot(gs[1, 1])
        axes[5] = fig.add_subplot(gs[1, 2])
        axes[6] = fig.add_subplot(gs[2, 2])

        feat_i = ellip_feat_extractor(
            img_i, mean_img, True, [axes[0], axes[1]], ['img_i', 'ellip_i'])
        feat_j = ellip_feat_extractor(
            img_j, mean_img, True, [axes[3], axes[4]], ['img_j', 'ellip_j'])
        
        n_digits = 2
        feat_i_disp = torch.round(feat_i.transpose(1, 0) * 10**n_digits) / (10**n_digits) 
        axes[2].text(0.1, 0.15, feat_i_disp)
        feat_j_disp = torch.round(feat_j.transpose(1, 0) * 10**n_digits) / (10**n_digits) 
        axes[5].text(0.1, 0.15, feat_j_disp)
        axes[5].set_title('ellip_feat_j')
        
        pose_ij = torch.round(pose_ij.transpose(1, 0) * 10**n_digits) / (10**n_digits) 
        axes[6].text(0.1, 0.2, pose_ij)
        axes[6].set_title('pose_ij')

        for ax in axes:
            if ax is not None:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.pause(1e-12)

        for ax in axes:
            if ax is not None:
                ax.remove()

def visualize_img_feat_tf(dataloader, mean_img):
    num_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)
    feats, poses = None, None
    fig = plt.figure(figsize=(24, 8))

    for batch_idx, data in enumerate(dataloader):

        img, pose = data

        nrows = 1
        ncols = 3
        gs = GridSpec(nrows, ncols, figure=fig)

        axes = [None] * (nrows*ncols)
        axes[0] = fig.add_subplot(gs[0, 0])
        axes[1] = fig.add_subplot(gs[0, 1])
        axes[2] = fig.add_subplot(gs[0, 2])

        feat = ellip_feat_extractor(img, mean_img, True, [axes[0], axes[1]], ['', ''])

        ori = pose[0, 2]
        sz_arw = 0.06
        (dx, dy) = (sz_arw * -math.sin(ori), sz_arw * math.cos(ori))
        axes[2].arrow(pose[0, 0]+dx, pose[0, 1]+dy, -dx, -dy,
                    head_width=5e-3, color="black", length_includes_head=True)
        circ_obj = Circle((0.0, 0.0), radius=0.088,
                            facecolor='None', edgecolor="grey", linestyle='-', linewidth=2, alpha=0.7)
        axes[2].add_patch(circ_obj)
        axes[2].plot(pose[0, 0], pose[0, 1], marker='o', color="red")
        axes[2].set_xlim([-0.15, 0.15])
        axes[2].set_ylim([-0.15, 0.15])

        for ax in axes:
            if ax is not None:
                ax.axis('off')

        plt.pause(1e-12)

        for ax in axes:
            if ax is not None:
                ax.remove()

def main():

    global DATASET_NAME

    # DATASET_NAME = "20200619_disc_6in_new_firmware"
    # DATASET_NAME = "20200624_pushing-6in-disc-straight-line"
    DATASET_NAME = "20200624_pushing-6in-disc-curves"

    srcdir_dataset = "{0}/local/datasets/{1}/train".format(
        BASE_PATH, DATASET_NAME)
    mean_img = cv2.imread(
        "{0}/local/digit/{1}/mean_img.png".format(BASE_PATH, DATASET_NAME))
    mean_img = mean_img.astype(np.uint8)

    transform = transforms.ToTensor()

    write_feats_file = False
    if write_feats_file:
        dataset = DigitImageTfDataset(srcdir_dataset, transform)
        write_feats_to_file(dataset, mean_img)

    vis_input_corr = False
    if vis_input_corr:
        dataset_pair = DigitImageTfPairsDataset(srcdir_dataset, transform)
        batch_size = min(len(dataset_pair), 2000)
        dataloader_pair = DataLoader(dataset_pair, batch_size=batch_size,
                                     shuffle=False, num_workers=1)
        network_input_correlations(dataloader_pair, mean_img)

    vis_feats_tfs_pairs = False
    if vis_feats_tfs_pairs:
        dataset_pair = DigitImageTfPairsDataset(srcdir_dataset, transform)
        batch_size = 1
        dataloader_pair = DataLoader(dataset_pair, batch_size=batch_size,
                                     shuffle=True, num_workers=1)
        visualize_img_feat_tf_pairs(dataloader_pair, mean_img)

    vis_feats_tfs = True
    if vis_feats_tfs:
        dataset = DigitImageTfDataset(srcdir_dataset, transform)
        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=1)
        visualize_img_feat_tf(dataloader, mean_img)

if __name__ == "__main__":

    main()
