# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import json

import numpy as np
from PIL import Image
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DigitImageTfPairsDataset(Dataset):
    """ Dataset (img1, pose1), (img2, pose2) for digit tactile images """

    def __init__(self, dir_dataset, transform=None, imgfeat_type=None, downsample_imgs=False, featmap_type=None, metadata=None, img_type='png'):

        self.dir_dataset = dir_dataset
        self.img_type = img_type
        self.imgfeat_type = imgfeat_type
        self.featmap_type = featmap_type

        self.transform = transform
        self.downsample_imgs = downsample_imgs
        self.metadata = metadata

        self.episodes = []
        self.img_idx_pairs = []
        self.pose_pairs = []

        self.min_num_data = 5

        # offsets for constructing idx pairs
        self.min_offset = 10
        self.max_offset = 50

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

            # loaded poses from json
            file_poses = "{0}/poses2d_episode_{1:04d}.json".format(
                dir_episode, episode_idx)
            data_json = None
            with open(file_poses) as f:
                data_json = json.load(f)
            poses = data_json['ee_poses2d__obj']

            # number of poses must match number of images in the episode
            assert(num_data == len(poses))
            print("[DigitImageTfPairsDataset] Loading dataset of ({0} images, {1} poses) for episode {2} from dir {3}".format(
                num_data, len(poses), episode_idx, dir_episode))

            # create dataset of image/pose pairs
            [curr_episodes, curr_img_idx_pairs,
                curr_pose_pairs] = self.create_dataset_pairs(episode_idx, poses)

            self.episodes.append(curr_episodes)
            self.img_idx_pairs.append(curr_img_idx_pairs)
            self.pose_pairs.append(curr_pose_pairs)

        # flatten into a single list
        self.episodes = [item for sublist in self.episodes for item in sublist]
        self.img_idx_pairs = [
            item for sublist in self.img_idx_pairs for item in sublist]
        self.pose_pairs = [
            item for sublist in self.pose_pairs for item in sublist]
    
    def create_dataset_pairs(self, episode_idx, poses):

        num_poses = len(poses)
        curr_episodes = []
        curr_img_idx_pairs = []
        curr_pose_pairs = []

        for i in range(self.skip_start_frames, num_poses-self.skip_end_frames):
            for j in range(i+1, num_poses-self.skip_end_frames):
                if (j < i + self.min_offset) | (j > i + self.max_offset):
                    continue
                curr_img_idx_pairs.append((i, j))
                curr_pose_pairs.append((poses[i], poses[j]))

        num_data_curr = len(curr_img_idx_pairs)
        curr_episodes = [episode_idx] * num_data_curr

        return (curr_episodes, curr_img_idx_pairs, curr_pose_pairs)

    @lru_cache(maxsize=10000)
    def get_image(self, episode, img_num):
        return Image.open("{0}/episode_{1:04d}/{2:04d}.{3}".format(self.dir_dataset, episode,
                                                                   img_num, self.img_type))

    @lru_cache(maxsize=10000)
    def get_image_feat(self, episode, img_num):
        if (self.imgfeat_type == "ellip"):
            feat = torch.load("{0}/episode_{1:04d}/{2:04d}_feat_ellip.pt".format(self.dir_dataset, episode, img_num))
            feat.squeeze_(0)

        elif (self.imgfeat_type == "keypoint"):
            feat = torch.load("{0}/episode_{1:04d}/{2:04d}_feat_keypoint.pt".format(self.dir_dataset, episode, img_num))
            feat.squeeze_(0)

        return feat

    @lru_cache(maxsize=10000)
    def get_image_featmap(self, episode, img_num):
        if (self.imgfeat_type == "keypoint"):
            featmap = torch.load("{0}/episode_{1:04d}/{2:04d}_featmap_keypoint.pt".format(self.dir_dataset, episode, img_num))

        return featmap

    def __getitem__(self, idx):

        # load image data
        img1 = self.get_image(self.episodes[idx], self.img_idx_pairs[idx][0])
        img2 = self.get_image(self.episodes[idx], self.img_idx_pairs[idx][1])

        # load pose data
        pose1 = torch.FloatTensor(self.pose_pairs[idx][0])
        pose2 = torch.FloatTensor(self.pose_pairs[idx][1])

        # load image feats
        if (self.imgfeat_type is not None):
            imgfeat1 = self.get_image_feat(
                self.episodes[idx], self.img_idx_pairs[idx][0])
            imgfeat2 = self.get_image_feat(
                self.episodes[idx], self.img_idx_pairs[idx][1])

        # load feat maps
        if (self.featmap_type is not None):
            featmap1 = self.get_image_featmap(
                self.episodes[idx], self.img_idx_pairs[idx][0])
            featmap2 = self.get_image_featmap(
                self.episodes[idx], self.img_idx_pairs[idx][1])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        if self.downsample_imgs:
            img1 = torch.nn.functional.interpolate(img1, size=64)
            img1 = torch.nn.functional.interpolate(img1.permute(0,2,1), size=64)
            img1 = img1.permute(0,2,1)

            img2 = torch.nn.functional.interpolate(img2, size=64)
            img2 = torch.nn.functional.interpolate(img2.permute(0,2,1), size=64)
            img2 = img2.permute(0,2,1)

        if any(elem is None for elem in [img1, img2, pose1, pose2]):
            print(
                "[DigitImageTfPairsDataset] Unable to read img, pose data at idx {0}".format(idx))
            return

        if (self.featmap_type is not None) & (self.metadata is not None) & (self.imgfeat_type is not None):
            data = (img1, img2, pose1, pose2, imgfeat1, imgfeat2, featmap1, featmap2, self.metadata)
        elif (self.metadata is not None) & (self.imgfeat_type is not None):
            data = (img1, img2, pose1, pose2, imgfeat1, imgfeat2, self.metadata)
        elif (self.imgfeat_type is not None):
            data = (img1, img2, pose1, pose2, imgfeat1, imgfeat2)
        else:
            data = (img1, img2, pose1, pose2)

        return data

    def __len__(self):
        return len(self.img_idx_pairs)
