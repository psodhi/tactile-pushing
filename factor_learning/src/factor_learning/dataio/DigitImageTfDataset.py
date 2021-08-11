# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import json

import numpy as np
from PIL import Image
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DigitImageTfDataset(Dataset):

    def __init__(self, dir_dataset, transform=None, imgfeat_type=None, downsample_imgs=False, img_type='png'):

        self.dir_dataset = dir_dataset
        self.img_type = img_type
        self.imgfeat_type = imgfeat_type

        self.transform = transform
        self.downsample_imgs = downsample_imgs

        self.episodes = []
        self.img_idxs = []
        self.poses = []

        self.min_num_data = 5

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
            print("[DigitImageTfDataset] Loading dataset of ({0} images, {1} poses) for episode {2} from dir {3}".format(
                num_data, len(curr_poses), episode_idx, dir_episode))

            # remove start/end set of frames
            self.remove_start_end_frames([curr_episodes, curr_img_idxs, curr_poses])
            
            self.episodes.append(curr_episodes)
            self.img_idxs.append(curr_img_idxs)
            self.poses.append(curr_poses)

        # flatten into a single list
        self.episodes = [item for sublist in self.episodes for item in sublist]
        self.img_idxs = [item for sublist in self.img_idxs for item in sublist]
        self.poses = [item for sublist in self.poses for item in sublist]
    
    def remove_start_end_frames(self, list_of_lists):
    
        for curr_list in list_of_lists:
            curr_list = curr_list[self.skip_start_frames:]
            curr_list = curr_list[:len(curr_list)-self.skip_end_frames]

    def get_item_loc(self, idx):

        loc = "{0}/episode_{1:04d}/{2:04d}".format(
            self.dir_dataset, self.episodes[idx], self.img_idxs[idx])

        return loc
    
    @lru_cache(maxsize=10000)
    def get_image(self, episode, img_num):
        return Image.open("{0}/episode_{1:04d}/{2:04d}.{3}".format(self.dir_dataset, episode,
                                                                   img_num, self.img_type))

    @lru_cache(maxsize=10000)
    def get_image_feat(self, episode, img_num):
        if (self.imgfeat_type == "ellip"):
            return torch.load("{0}/episode_{1:04d}/{2:04d}_feat_ellip.pt".format(self.dir_dataset, episode, img_num))
        elif (self.imgfeat_type == "keypoint"):
            return torch.load("{0}/episode_{1:04d}/{2:04d}_feat_keypoint.pt".format(self.dir_dataset, episode, img_num))
    
    def __getitem__(self, idx):

        # load image data
        img = self.get_image(self.episodes[idx], self.img_idxs[idx])

        # load manual image feats
        if (self.imgfeat_type is not None):
            imgfeat = self.get_image_feat(
                self.episodes[idx], self.img_idxs[idx])

            imgfeat.squeeze_(0)

        # load pose data
        pose = torch.FloatTensor(self.poses[idx])

        if self.transform is not None:
            img = self.transform(img)
        
        if self.downsample_imgs:
            img = torch.nn.functional.interpolate(img, size=64)
            img = torch.nn.functional.interpolate(img.permute(0,2,1), size=64)
            img = img.permute(0,2,1)

        if any(elem is None for elem in [img, pose]):
            print(
                "[DigitImageTfDataset] Unable to read img, pose data at idx {0}".format(idx))
            return
        
        if self.imgfeat_type is not None:
            data = (img, pose, imgfeat)
        else:
            data = (img, pose)

        return data

    def __len__(self):
        return len(self.img_idxs)
