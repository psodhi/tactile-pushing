# Copyright (c) Facebook, Inc. and its affiliates.

from factor_learning.keypoint_prediction.train_utils import _flatten_, _unflatten_, switch_majors, load_checkpoint
from factor_learning.keypoint_prediction.keypoint_detector import KeypointDetector
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


def _to_img_coord(x, vis_scaling_factor=1):
    return ((x * 4 + 32) * vis_scaling_factor).astype(np.uint8)


def to_img_coords(x, y, vis_scaling_factor=1):
    return (_to_img_coord(x, vis_scaling_factor), _to_img_coord(y, vis_scaling_factor))


def draw_keypoints(imgs, keypoints, vis_scaling_factor=1):
    '''
    imgs: T x B x C x H x W
    keypoints: T x B x Nkp
    '''

    T, B, C, H, W = imgs.shape

    # B = 1
    kps_x = keypoints.x.cpu().detach().squeeze(1).numpy()
    kps_y = keypoints.y.cpu().detach().squeeze(1).numpy()
    (kps_x, kps_y) = to_img_coords(kps_x, kps_y, vis_scaling_factor)

    imgs_copy = imgs.squeeze(1).cpu().detach().numpy()

    imgs_kps = torch.zeros(imgs_copy.shape)
    for img_idx, img_copy in enumerate(imgs_copy):

        for kp_idx in range(kps_x.shape[1]):
            img_kp = img_copy.transpose(1, 2, 0)
            img_kp = img_kp * 255

            x = kps_x[img_idx, kp_idx]
            y = kps_y[img_idx, kp_idx]
            img_kp = cv2.circle(cv2.UMat(img_kp), (x, y),
                                radius=4, color=(0, 0, 255), thickness=-1)

            img_kp = torch.from_numpy(
                img_kp.get().transpose(2, 0, 1) / 255).float()

        imgs_kps[img_idx, :, :, :] = img_kp

    imgs_kps = _unflatten_(imgs_kps, T)

    return imgs_kps


def load_keypoint_model(cfg, device, eval_mode=True):
    path = "{0}/local/{1}/{2}.pt".format(BASE_PATH,
                                         cfg.checkpoint.path, cfg.eval.model_name)
    # path = "{0}/local/models/keypoint_prediction/{1}.pt".format(
        # BASE_PATH, cfg.eval.model_name)
    checkpoint = load_checkpoint(path, device)
    model = KeypointDetector(**cfg.keypoint_detector)
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_mode:
        model.eval()

    return model


def rescale_img(img, vis_scaling_factor):
    img = img.squeeze()
    img = torch.nn.functional.interpolate(img, size=64*vis_scaling_factor)
    img = torch.nn.functional.interpolate(
        img.permute(0, 2, 1), size=64*vis_scaling_factor)
    img = img.permute(0, 2, 1)
    img = img[None, None, :]

    return img


def keypoint_feat_extractor(img, keypoint_detector, visualize_feats=False, rgb2bgr=False):

    # img input: (T, B, C, H, W), keypoint output: (T x B x Nkp)
    img = img[None, None, :]

    if (rgb2bgr):
        keypoints, prob_map, feat_map = keypoint_detector.encode(img[:, :, [2, 1, 0], :, :])
    else:
        keypoints, prob_map, feat_map = keypoint_detector.encode(img)
    
    kps_x = torch.div(torch.add(torch.mul(keypoints.x.squeeze(0), 4), 32), 64)
    kps_y = torch.div(torch.add(torch.mul(keypoints.y.squeeze(0), 4), 32), 64)
    feat = torch.cat([kps_x, kps_y], 1)

    if visualize_feats:
        vis_scaling_factor = 3
        img = rescale_img(img, vis_scaling_factor)
        img_kps = draw_keypoints(img, keypoints, vis_scaling_factor)

        img_kps_vis = transforms.ToPILImage()(img_kps.squeeze())
        plt.imshow(img_kps_vis)
        plt.pause(1e-12)
        plt.clf()

    return feat, prob_map, feat_map


def write_feats_to_file(dataset, cfg):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    keypoint_detector = load_keypoint_model(cfg, device)

    num_data = len(dataset)
    for idx in range(0, num_data):

        img, pose = dataset[idx]
        feat, prob_map, feat_map = keypoint_feat_extractor(
            img, keypoint_detector, visualize_feats=False, rgb2bgr=cfg.dataloader.rgb2bgr)

        item_loc = dataset.get_item_loc(idx)
        if True:
            dstfile = "{0}_feat_keypoint.pt".format(item_loc)
            torch.save(feat.data, dstfile)
            print("Writing features to: {0}".format(dstfile))
        if False:
            feat_map.squeeze_(0).squeeze_(0)
            dstfile = "{0}_featmap_keypoint.pt".format(item_loc)
            torch.save(feat_map.data, dstfile)
            print("Writing featmaps to: {0}".format(dstfile))


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(BASE_PATH, "config/keypoint_detector.yaml")


@hydra.main(config_path=config_path, strict=False)
def main(cfg):

    srcdir_dataset = "{0}/local/datasets/{1}/".format(
        BASE_PATH, cfg.eval.dataset_name)
    subdirs = next(os.walk(srcdir_dataset))[1]
        
    for subdir in subdirs:
        print("{0}/{1}".format(srcdir_dataset, subdir))
        transform = transforms.ToTensor()
        dataset = DigitImageTfDataset(
            "{0}/{1}".format(srcdir_dataset, subdir), transform=transform, downsample_imgs=cfg.dataloader.downsample_imgs)
        write_feats_to_file(dataset, cfg)

if __name__ == "__main__":

    main()
