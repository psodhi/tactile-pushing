# Copyright (c) Facebook, Inc. and its affiliates.

import os
import json
import glob

import numpy as np
from PIL import Image
import torch

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class DigitImageTFDatasetJsonWriter:
    def __init__(self, dataset_name, srcdir_dataset, dataset_type='', imgfeat_type='', img_type='png'):

        self.dataset_name = dataset_name
        self.srcdir_dataset = srcdir_dataset
        self.img_type = img_type

        self.imgfeat_type = imgfeat_type
        self.dataset_type = dataset_type

        # to be logged data
        self.imgfeat_list = []
        self.obj_pose2d_list = []
        self.ee_pose2d_list = []
        self.contact_episode_list = []
        self.contact_flag_list = []

        self.init_dataset_params()

        self.dstdir_json = self.srcdir_dataset

    def init_dataset_params(self):
        self.params = {}
        self.params['obj_radius'] = 0.088

    def save_data2d_json(self):

        data = {'params': self.params,
                'img_feats': self.imgfeat_list,
                'obj_poses_2d': self.obj_pose2d_list,
                'ee_poses_2d': self.ee_pose2d_list,
                'contact_flag': self.contact_flag_list,
                'contact_episode': self.contact_episode_list}

        dstfile = "{0}/{1}-{2}-{3}.json".format(
            self.dstdir_json, self.dataset_name, self.dataset_type, self.imgfeat_type)
        with open(dstfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        print("Wrote json dataset for episodes {0} at:\n {1} ".format(
            set(self.contact_episode_list), dstfile))

    def remove_start_end_frames(self, list_of_lists):

        for curr_list in list_of_lists:
            curr_list = curr_list[self.skip_start_frames:]
            curr_list = curr_list[:len(curr_list)-self.skip_end_frames]

    def get_image(self, episode, img_num):
        return Image.open("{0}/episode_{1:04d}/{2:04d}.{3}".format(self.dir_dataset, episode,
                                                                   img_num, self.img_type))

    def get_image_feat(self, episode, img_num):
        if (self.imgfeat_type == "ellip"):
            return torch.load("{0}/episode_{1:04d}/{2:04d}_feat_ellip.pt".format(self.srcdir_dataset, episode, img_num))
        elif (self.imgfeat_type == "keypoint"):
            return torch.load("{0}/episode_{1:04d}/{2:04d}_feat_keypoint.pt".format(self.srcdir_dataset, episode, img_num))

    def write(self):

        self.min_num_data = 5

        # reduce outliers due to noisy feature estimates at start/end of episode
        self.skip_start_frames = 2
        self.skip_end_frames = 2

        # collect img, pose pairs for each episode and concatenate as lists
        subdirs = sorted(glob.glob("{0}/*".format(self.srcdir_dataset)))
        for (curr_idx, dir_episode) in enumerate(subdirs):

            num_imgs = len(
                glob.glob("{0}/*.{1}".format(dir_episode, self.img_type)))
            if (num_imgs < self.min_num_data):
                continue

            episode_idx = int((dir_episode.split('/')[-1]).split('_')[-1])

            file_poses = "{0}/poses2d_episode_{1:04d}.json".format(
                dir_episode, episode_idx)
            data_json = None
            with open(file_poses) as f:
                data_json = json.load(f)
            num_imgs_curr = len(
                glob.glob("{0}/*.{1}".format(dir_episode, self.img_type)))

            # collect data to be logged from current episode
            curr_imgfeats = []
            for img_idx in range(0, num_imgs_curr):
                imgfeat = (self.get_image_feat(
                    episode_idx, img_idx).data.numpy().squeeze(0)).tolist()
                curr_imgfeats.append(imgfeat)
            curr_obj_poses2d = data_json['obj_poses2d__world']
            curr_ee_poses2d = data_json['ee_poses2d__world']
            curr_episodes = [episode_idx] * num_imgs_curr
            curr_contact_flags = [1] * num_imgs_curr

            # number of poses must match number of images in the episode
            assert(num_imgs_curr == len(curr_obj_poses2d)
                   == len(curr_ee_poses2d))
            print("[DigitImageTFDatasetJsonWriter] Writing dataset of ({0} images, {1} poses) for episode {2} from dir {3}".format(
                num_imgs_curr, len(curr_obj_poses2d), episode_idx, dir_episode))

            # remove start/end set of frames
            self.remove_start_end_frames(
                [curr_imgfeats, curr_obj_poses2d, curr_ee_poses2d, curr_episodes, curr_contact_flags])

            self.imgfeat_list.append(curr_imgfeats)
            self.obj_pose2d_list.append(curr_obj_poses2d)
            self.ee_pose2d_list.append(curr_ee_poses2d)
            self.contact_episode_list.append(curr_episodes)
            self.contact_flag_list.append(curr_contact_flags)

            self.contact_episode_idx = episode_idx

        # flatten into a single list
        self.imgfeat_list = [
            item for sublist in self.imgfeat_list for item in sublist]
        self.obj_pose2d_list = [
            item for sublist in self.obj_pose2d_list for item in sublist]
        self.ee_pose2d_list = [
            item for sublist in self.ee_pose2d_list for item in sublist]
        self.contact_episode_list = [
            item for sublist in self.contact_episode_list for item in sublist]
        self.contact_flag_list = [
            item for sublist in self.contact_flag_list for item in sublist]

        self.save_data2d_json()


def main():
    # straight-line, curves, trial-1
    dataset_name = "20200624_pushing-6in-disc-curves"
    # dataset_name = "20200824_rectangle-pushing-1" # 1, 2, 3
    # dataset_name = "20200928_rectangle-pushing-edges" # *-edges, *-corners
    # dataset_name = "20200928_ellipse-pushing" # *-straight, *-

    dataset_type = "all"  # train, test, all
    imgfeat_type = "keypoint"  # keypoint, ellip

    srcdir_dataset = "{0}/local/datasets/{1}/{2}".format(
        BASE_PATH, dataset_name, dataset_type)
    img_tf_writer_json = DigitImageTFDatasetJsonWriter(
        dataset_name, srcdir_dataset, dataset_type=dataset_type, imgfeat_type=imgfeat_type)
    img_tf_writer_json.write()


if __name__ == '__main__':
    main()
