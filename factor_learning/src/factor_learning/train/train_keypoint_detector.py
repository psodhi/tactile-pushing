# Copyright (c) Facebook, Inc. and its affiliates.

from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.utils
import torch.nn as nn
import torch.optim as optim
import torch

import hydra
import numpy as np
import math
import os

import cv2
from PIL import Image
from factor_learning.dataio.DigitImageTfSeqDataset import DigitImageTfSeqDataset

from factor_learning.keypoint_prediction.keypoint_detector import KeypointDetector
from factor_learning.keypoint_prediction.train_utils import _flatten_, _unflatten_, switch_majors, save_checkpoint

from factor_learning.utils import utils

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(BASE_PATH, "config/keypoint_detector.yaml")


def _to_img_coord(x):
    vis_scaling_factor = 1
    return ((x * 4 + 32) * vis_scaling_factor).astype(np.uint8)


def to_img_coords(x, y):
    return (_to_img_coord(x), _to_img_coord(y))


def draw_keypoints(imgs, keypoints):
    '''
    imgs: T x B x C x H x W
    keypoints: T x B x Nkp
    '''

    T, B, C, H, W = imgs.shape

    # B = 1
    kps_x = keypoints.x.cpu().detach().squeeze(1).numpy()
    kps_y = keypoints.y.cpu().detach().squeeze(1).numpy()
    (kps_x, kps_y) = to_img_coords(kps_x, kps_y)

    imgs_copy = imgs.squeeze(1).cpu().detach().numpy()

    imgs_kps = torch.zeros(imgs_copy.shape)
    for img_idx, img_copy in enumerate(imgs_copy):

        img_kp = img_copy.transpose(1, 2, 0)
        img_kp = img_kp * 255

        for kp_idx in range(kps_x.shape[1]):
            x = kps_x[img_idx, kp_idx]
            y = kps_y[img_idx, kp_idx]
            img_kp = cv2.circle(cv2.UMat(img_kp), (x, y),
                                radius=2, color=(255, 0, 0), thickness=-1)

        img_kp = torch.from_numpy(
            img_kp.get().transpose(2, 0, 1) / 255).float()
        imgs_kps[img_idx, :, :, :] = img_kp

    imgs_kps = _unflatten_(imgs_kps, T)

    return imgs_kps


def load_input_data(cfg):
    # input data transforms
    if (cfg.transforms.normalize_img_inputs):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            cfg.transforms.norm_mean_img, cfg.transforms.norm_std_img)])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train_datasets = []
    test_datasets = []
    for dataset_name in cfg.dataset_names:
        srcdir_dataset = "{0}/local/datasets/{1}".format(
            BASE_PATH, dataset_name)
        train_datasets.append(DigitImageTfSeqDataset(
            srcdir_dataset + "/train-keypt", transform=transform, downsample_imgs=cfg.dataloader.downsample_imgs))
        test_datasets.append(DigitImageTfSeqDataset(
            srcdir_dataset + "/test", transform=transform, downsample_imgs=cfg.dataloader.downsample_imgs))
    
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataloader.batch_size, drop_last=False,
                                  shuffle=cfg.dataloader.shuffle, num_workers=cfg.dataloader.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.dataloader.batch_size, drop_last=False,
                                 shuffle=cfg.dataloader.shuffle, num_workers=cfg.dataloader.num_workers)

    return (train_dataset, train_dataloader, test_dataset, test_dataloader)

def load_no_contact_image(cfg):
    # input data transforms
    if (cfg.transforms.normalize_img_inputs):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            cfg.transforms.norm_mean_img, cfg.transforms.norm_std_img)])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    no_contact_img = Image.open(
        "{0}/local/digit/{1}/mean_img.png".format(BASE_PATH, cfg.dataset_names[0]))
    no_contact_img = transform(no_contact_img)
    if (cfg.dataloader.downsample_imgs):
        no_contact_img = torch.nn.functional.interpolate(
            no_contact_img, size=64)
        no_contact_img = torch.nn.functional.interpolate(
            no_contact_img.permute(0, 2, 1), size=64)
        no_contact_img = no_contact_img.permute(0, 2, 1)
        
    return no_contact_img

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

    # initialize visualizer
    vis_train = cfg.train.visualize
    vis_interval = cfg.train.vis_interval
    tb_dir = "{0}".format(prefix)
    os.system('mkdir -p {0}/runs/{1}'.format(BASE_PATH, tb_dir))
    writer = SummaryWriter("{0}/runs/{1}".format(BASE_PATH, tb_dir))

    # load dataset
    [train_dataset, train_dataloader, test_dataset, test_dataloader]= load_input_data(cfg)

    # load no contact image
    if (cfg.dataloader.use_no_contact_img):
        no_contact_img = load_no_contact_image(cfg)
        no_contact_img = no_contact_img.to(device)
    
    # initialize model and optimizer
    model = KeypointDetector(**cfg.keypoint_detector)
    model = model.to(device)

    weight_decay = cfg.train.reg_weight if (
        cfg.train.reg_type == "l2") else 0.0
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=weight_decay)

    # initialize data variables
    loss = np.Inf
    num_batches = math.ceil(len(train_dataset) / cfg.dataloader.batch_size)
    loss_vec_train = torch.zeros(num_batches, 1)
    loss_vec_test = torch.zeros(math.ceil(len(test_dataset) / cfg.dataloader.batch_size), 1)

    # save model network
    save_model = cfg.checkpoint.enable_save
    save_model_interval = cfg.checkpoint.save_interval_epoch
    if (cfg.checkpoint.path_abs):
        path_model = cfg.checkpoint.path
    else:
        path_model = "{0}/local/{1}".format(BASE_PATH, cfg.checkpoint.path)

    num_epochs = cfg.train.epochs
    for epoch in range(num_epochs):

        if ((save_model) & (epoch % save_model_interval == 0)):
            model_name = "{0}_keypoint_detector_model_epoch{1}.pt".format(
                prefix, epoch)
            save_path = "{0}/{1}".format(path_model, model_name)
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=torch.mean(
                loss_vec_train), itr=0, path=save_path)

        for train_idx, data in enumerate(train_dataloader):
            # read model input
            imgs, poses = data
            imgs = switch_majors(imgs)  # (T, B, C, H, W)
            imgs = imgs.to(device)

            # call model and optimize
            if (cfg.dataloader.rgb2bgr):
                loss, reconstruct_imgs, keypoints = model.forward(imgs[:, :, [2, 1, 0], :, :])
                reconstruct_imgs = reconstruct_imgs[:, :, [2, 1, 0], :, :]
            else:
                loss, reconstruct_imgs, keypoints = model.forward(imgs)

            total_loss = loss.total
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # display loss
            loss_vec_train[train_idx] = total_loss.item()
            writer.add_scalar("train/seq/loss",
                              loss_vec_train[train_idx], train_idx + epoch * len(loss_vec_train))

            print("epoch {0}/{1}, seq: {2}/{3}, total_loss: {4}".format(epoch,
                                                                        num_epochs-1, train_idx, num_batches, total_loss))
            # display images
            if ((vis_train) & (train_idx % vis_interval == 0)):
                keypoint_imgs = draw_keypoints(imgs, keypoints)

                reconstruct_imgs_disp = _flatten_(switch_majors(reconstruct_imgs))
                if cfg.transforms.normalize_img_inputs:
                    reconstruct_imgs_disp = utils.denormalize_img(
                        reconstruct_imgs_disp, cfg.transforms.norm_mean_img, cfg.transforms.norm_std_img)
                grid = torchvision.utils.make_grid(reconstruct_imgs_disp)
                writer.add_image("reconstruct_imgs", grid, train_idx)

                keypoint_imgs_disp = _flatten_(switch_majors(keypoint_imgs))
                if cfg.transforms.normalize_img_inputs:
                    keypoint_imgs_disp = utils.denormalize_img(
                        keypoint_imgs_disp, cfg.transforms.norm_mean_img, cfg.transforms.norm_std_img)

                grid = torchvision.utils.make_grid(keypoint_imgs_disp)
                writer.add_image("keypoint_imgs", grid, train_idx)

        writer.add_scalar("train/epoch/loss", torch.mean(loss_vec_train), epoch)

        if (cfg.train.validation):
            for batch_idx, data in enumerate(test_dataloader):

                # read model input
                imgs, poses = data
                imgs = switch_majors(imgs)  # (T, B, C, H, W)
                imgs = imgs.to(device)

                # call model
                loss, reconstruct_imgs, keypoints = model.forward(imgs)
                total_loss = loss.total

                loss_vec_test[batch_idx] = total_loss.item()

            print("epoch {0}/{1}, test loss: {2}".format(epoch,
                                                        num_epochs-1, torch.mean(loss_vec_test)))
            writer.add_scalar("test/epoch/loss", torch.mean(loss_vec_test), epoch)
        
if __name__ == '__main__':
    main()
