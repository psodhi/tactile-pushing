# Copyright (c) Facebook, Inc. and its affiliates.

from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.utils
import torch.nn as nn
import torch.optim as optim
import torch
import cv2
import hydra
import numpy as np
import os

from PIL import Image
import imageio
from attrdict import AttrDict

from factor_learning.dataio.DigitImageTfDataset import DigitImageTfDataset
from factor_learning.keypoint_prediction.keypoint_detector import KeypointDetector
from factor_learning.keypoint_prediction.train_utils import _flatten_, _unflatten_, switch_majors, load_checkpoint

from factor_learning.utils import utils

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(BASE_PATH, "config/keypoint_detector.yaml")


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
                                radius=4, color=(0,0,255), thickness=-1)

            img_kp = torch.from_numpy(
                img_kp.get().transpose(2, 0, 1) / 255).float()
                
        imgs_kps[img_idx, :, :, :] = img_kp

    imgs_kps = _unflatten_(imgs_kps, T)

    return imgs_kps

def load_model(cfg, device, eval_mode=True):
    path = "{0}/local/{1}/{2}.pt".format(BASE_PATH,
                                         cfg.checkpoint.path, cfg.eval.model_name)
    # path = "{0}/local/models/keypoint_prediction/{1}.pt".format(BASE_PATH, cfg.eval.model_name)
    checkpoint = load_checkpoint(path, device)
    model = KeypointDetector(**cfg.keypoint_detector)
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_mode:
        model.eval()

    return model

def rescale_img(img, cfg):
    img = (img.squeeze(0)).squeeze(0)
    img = torch.nn.functional.interpolate(img, size=64*cfg.eval.vis_scaling_factor)
    img = torch.nn.functional.interpolate(img.permute(0,2,1), size=64*cfg.eval.vis_scaling_factor)
    img = img.permute(0,2,1)
    img = img[None, None, :]

    return img

def init_img_list():
    img_list = AttrDict()
    img_list.orig = []
    img_list.feature = []
    img_list.prob = []
    img_list.keypoint = []
    img_list.gblob = []
    img_list.recon = []

    return img_list

def save_img_list_video(img_list, dstdir, dataset_name, fps=15):

    # kargs = { 'fps': 3, 'quality': 10, 'macro_block_size': None, 
    # 'ffmpeg_params': ['-s','600x450'] }

    print("Writing img list outputs as videos to: {0}".format(dstdir))

    vidfile = "{0}/orig_{1}.mp4".format(dstdir, dataset_name)
    imageio.mimwrite(vidfile, img_list.orig, fps=fps)
    vidfile = "{0}/feature_{1}.mp4".format(dstdir, dataset_name)
    imageio.mimwrite(vidfile, img_list.feature, fps=fps)
    vidfile = "{0}/prob_{1}.mp4".format(dstdir, dataset_name)
    imageio.mimwrite(vidfile, img_list.prob, fps=fps)
    vidfile = "{0}/keypoint_{1}.mp4".format(dstdir, dataset_name)
    imageio.mimwrite(vidfile, img_list.keypoint, fps=fps)
    vidfile = "{0}/gblob_{1}.mp4".format(dstdir, dataset_name)
    imageio.mimwrite(vidfile, img_list.gblob, fps=fps)
    vidfile = "{0}/recon_{1}.mp4".format(dstdir, dataset_name)
    imageio.mimwrite(vidfile, img_list.recon, fps=fps)

@hydra.main(config_path=config_path, strict=False)
def main(cfg):

    # prefix for loading checkpoint, saving tb dir, cfg file
    prefix = cfg.prefix + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    # print/save config params
    print(cfg.pretty())
    with open("{0}/runs/{1}.txt".format(BASE_PATH, prefix), "w") as f:
        print(cfg.pretty(), file=f)

    # detect device: cpu/gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {0}".format(device))

    # random seed
    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)

    # input data transforms
    if (cfg.transforms.normalize_img_inputs):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            cfg.transforms.norm_mean_img, cfg.transforms.norm_std_img)])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # load dataset
    srcdir_dataset = "{0}/local/datasets/{1}".format(
        BASE_PATH, cfg.eval.dataset_name)
    test_dataset = DigitImageTfDataset(
        srcdir_dataset + "/test", transform, downsample_imgs=cfg.dataloader.downsample_imgs)

    # load model
    model = load_model(cfg, device)

    # write img list to videos
    img_list = init_img_list()
    dstdir = "{0}/local/visualizations/".format(BASE_PATH)

    for (idx, data) in enumerate(test_dataset):

        img, poses = data
        img = img[None, None, :]  # T x B x C x H x W

        if (cfg.dataloader.rgb2bgr):
            keypoints, prob_map, feature_map = model.encode(img[:, :, [2, 1, 0], :, :])
        else:
            keypoints, prob_map, feature_map = model.encode(img)
        
        recon_images, uf_gaussian_blobs = model.decode(keypoints)
        if (cfg.dataloader.rgb2bgr):
            recon_images = recon_images[:, :, [2, 1, 0], :, :]

        img = rescale_img(img, cfg)

        img_kps = draw_keypoints(img, keypoints, cfg.eval.vis_scaling_factor)        
        img_kps = (img_kps.squeeze(0)).squeeze(0)

        # normalize to [0, 1]
        feature_map_norm = ((feature_map.squeeze(0)).squeeze(0)) / torch.max((feature_map.squeeze(0)).squeeze(0))
        prob_map_norm = ((prob_map.squeeze(0)).squeeze(0)) / torch.max((prob_map.squeeze(0)).squeeze(0))
        gblob_norm = ((uf_gaussian_blobs.squeeze(0)).squeeze(0)) / torch.max((uf_gaussian_blobs.squeeze(0)).squeeze(0))

        # save to list
        img_list['orig'].append(transforms.ToPILImage()((img.squeeze((0)).squeeze(0))))
        img_list['feature'].append(transforms.ToPILImage()(feature_map_norm))
        img_list['prob'].append(transforms.ToPILImage()(prob_map_norm))
        img_list['keypoint'].append(transforms.ToPILImage()(img_kps))
        img_list['gblob'].append(transforms.ToPILImage()(gblob_norm))
        img_list['recon'].append(transforms.ToPILImage()((recon_images.squeeze(0)).squeeze(0)))

    save_img_list_video(img_list, dstdir, cfg.eval.dataset_name)


if __name__ == '__main__':
    main()
