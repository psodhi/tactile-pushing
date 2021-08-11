# Copyright (c) Facebook, Inc. and its affiliates.

from factor_learning.utils import utils
from factor_learning.dataio.DigitImageTfDataset import DigitImageTfDataset
from factor_learning.dataio.DigitImageTfPairsDataset import DigitImageTfPairsDataset

import os
import numpy as np
import math

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

import seaborn as sns
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
plt.rcParams.update({'font.size': 22})

def visualize_correlation(feat_ij, pose_ij):

    data_tensor = torch.cat([feat_ij, pose_ij], 1)
    data = data_tensor.data.numpy()

    data_frame = pd.DataFrame(data)
    if (IMGFEAT_TYPE == "ellip"):
        data_frame.columns = ['$f_{ij}[0]$: cx', '$f_{ij}[1]$: cy', '$f_{ij}[2]$: szx',
                            '$f_{ij}$[3]: szy', '$f_{ij}[4]$: c$\\alpha$', '$f_{ij}[5]$: s$\\alpha$',
                            '$T_{ij}[0]$: tx', '$T_{ij}[1]$: ty', '$T_{ij}[2]$: c$\\theta$', '$T_{ij}[3]$: s$\\theta$']
    elif (IMGFEAT_TYPE == "keypoint"):
        data_frame.columns = ['$f_{ij}[0]$: cx', '$f_{ij}[1]$: cy', 
                            '$T_{ij}[0]$: tx', '$T_{ij}[1]$: ty', '$T_{ij}[2]$: c$\\theta$', '$T_{ij}[3]$: s$\\theta$']

    corr_matrix = data_frame.corr()

    # plot correlation scatter plot
    fig1 = plt.figure()
    scatter_matrix(data_frame, figsize=(18, 18))

    save_fig = True
    if save_fig:
        dstdir = "{0}/local/visualizations/correlations/{1}".format(BASE_PATH, DATASET_NAME)
        utils.make_dir(dstdir, False)
        plt.savefig("{0}/{1}_episode_{2:03d}.png".format(dstdir, IMGFEAT_TYPE, EPISODE_NUM))
        print("{0}/{1}_episode_{2:03d}.png".format(dstdir, IMGFEAT_TYPE, EPISODE_NUM))

    # plt.show()

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

    # plt.show(block=True)

def network_input_correlations(dataloader):

    num_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)
    feats, poses = None, None
    for batch_idx, data in enumerate(dataloader):

        print("[network_input_correlations] batch {0} / {1} of size {2}".format(
            batch_idx, num_batches-1, dataloader.batch_size))

        img_i, img_j, pose_i, pose_j, feat_i, feat_j = data

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

def main():

    global DATASET_NAME, IMGFEAT_TYPE, EPISODE_NUM

    # DATASET_NAME = "20200619_disc_6in_new_firmware"
    # DATASET_NAME = "20200624_pushing-6in-disc-straight-line"
    # DATASET_NAME = "20200624_pushing-6in-disc-curves"
    # DATASET_NAME = "20200624_pushing-6in-disc-trial1"

    # DATASET_NAME = "20200928_rectangle-pushing-edges"
    # DATASET_NAME = "20200928_rectangle-pushing-corners"

    DATASET_NAME = "20200928_ellipse-pushing-straight"
    # DATASET_NAME = "20200928_ellipse-pushing"

    # IMGFEAT_TYPE = "ellip"
    IMGFEAT_TYPE = "keypoint"

    transform = transforms.ToTensor()
    for EPISODE_NUM in range(0, 25):
        srcdir_dataset = "{0}/local/datasets/{1}/individual/{2}".format(
            BASE_PATH, DATASET_NAME, EPISODE_NUM)
                
        dataset_pair = DigitImageTfPairsDataset(srcdir_dataset, transform=transform, imgfeat_type=IMGFEAT_TYPE)
        batch_size = min(len(dataset_pair), 2000)
        if (batch_size < 10):
            continue

        dataloader_pair = DataLoader(dataset_pair, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
        network_input_correlations(dataloader_pair)

if __name__ == "__main__":

    main()
