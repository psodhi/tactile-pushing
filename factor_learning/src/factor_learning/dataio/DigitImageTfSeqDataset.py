# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import json

import numpy as np
from PIL import Image
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DigitImageTfSeqDataset(Dataset):

    def __init__(self, dir_dataset, transform=None, downsample_imgs=False, img_type='png'):

        self.dir_dataset = dir_dataset
        self.img_type = img_type

        self.transform = transform
        self.downsample_imgs = downsample_imgs

        self.episodes = []
        self.img_idxs = []
        self.poses = []

        self.min_num_data = 3

        # reduce outliers due to noisy ellip feature estimates at start/end of episode
        self.skip_start_frames = 2
        self.skip_end_frames = 2

        # collect img, pose pairs for each episode and concatenate as lists
        subdirs = sorted(glob.glob("{0}/*".format(dir_dataset)))
        for (curr_idx, dir_episode) in enumerate(subdirs):

            num_data = len(
                glob.glob("{0}/*.{1}".format(dir_episode, img_type)))
            if (num_data < self.min_num_data):
                continue

            episode_idx = int((dir_episode.split('/')[-1]).split('_')[-1])

            # load poses, img_idxs
            file_poses = "{0}/poses2d_episode_{1:04d}.json".format(
                dir_episode, episode_idx)
            data_json = None
            with open(file_poses) as f:
                data_json = json.load(f)
            num_imgs_curr = len(
                glob.glob("{0}/*.{1}".format(dir_episode, self.img_type)))

            curr_episodes = [episode_idx] * num_imgs_curr
            curr_img_idxs = list(range(0, num_imgs_curr))
            curr_poses = data_json['ee_poses2d__obj']

            # number of poses must match number of images in the episode
            assert(num_data == len(curr_poses))
            print("[DigitImageTfSeqDataset] Loading dataset of ({0} images, {1} poses) for episode {2} from dir {3}".format(
                num_data, len(curr_poses), episode_idx, dir_episode))

            # remove start/end set of frames
            self.remove_start_end_frames([curr_episodes, curr_img_idxs, curr_poses])
            
            self.episodes.append(curr_episodes)
            self.img_idxs.append(curr_img_idxs)
            self.poses.append(curr_poses)
    
    def remove_start_end_frames(self, list_of_lists):
    
        for curr_list in list_of_lists:
            curr_list = curr_list[self.skip_start_frames:]
            curr_list = curr_list[:len(curr_list)-self.skip_end_frames]

    def get_item_loc(self, idx):

        loc = "{0}/episode_{1:04d}/{2:04d}".format(
            self.dir_dataset, self.episodes[idx], self.img_idxs[idx])

        return loc
    
    @lru_cache(maxsize=10000)
    def get_images_seq(self, seq_idx):
        episodes_seq = self.episodes[seq_idx]
        img_idxs_seq = self.img_idxs[seq_idx]
        imgs = []
        for (idx, img_idx) in enumerate(img_idxs_seq):
            imgs.append(Image.open("{0}/episode_{1:04d}/{2:04d}.{3}".format(self.dir_dataset, episodes_seq[idx],
                                                                   img_idx, self.img_type)))
        return imgs

    def __getitem__(self, seq_idx):

        # load image data
        imgs_list = self.get_images_seq(seq_idx)

        # load pose data
        poses = torch.FloatTensor(self.poses[seq_idx])

        if self.transform is not None:
            for idx, img in enumerate(imgs_list):
                imgs_list[idx] = self.transform(img)

        if self.downsample_imgs:
            for idx, img in enumerate(imgs_list):
                img = torch.nn.functional.interpolate(img, size=64)
                img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=64)
                img = img.permute(0, 2, 1)

                imgs_list[idx] = img

        if any(elem is None for elem in [imgs_list, poses]):
            print(
                "[DigitImageTfSeqDataset] Unable to read img, pose data at seq_idx {0}".format(seq_idx))
            return
        
        imgs = torch.stack(imgs_list)
        data = (imgs, poses)

        return data

    def __len__(self):
        return len(self.img_idxs)
