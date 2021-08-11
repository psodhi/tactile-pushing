#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import math
import json
import hydra
import os
import time
import pickle as pkl

import pushestcpp
import gtsam

from itertools import combinations, permutations, combinations_with_replacement
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import pushestpy.dataio.data_process as data_process
import pushestpy.utils as utils
import pushestpy.eval.qual_visualizer as qviz

from pushestpy.utils.Logger import Logger

class RealPushGraphOptimizeISAM2():
    def __init__(self, cfg):

        # params
        self.cfg = cfg
        self.ee_radius = 0

        # dataio
        self.torch_model_file = "{0}/local/models/{1}.pt".format(
            BASE_PATH, self.cfg.dataio.torch_model_name)
        self.push_data = {}
        self.push_data_file = {}
        self.load_push_dataset()

        # load planar sdf object
        self.object_sdf = None
        self.load_object_sdf()

        # factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.init_vals = gtsam.Values()

        # optimizer
        self.init_optimizer()
        self.init_noise_models()

        # logger, visualizer objects
        params_logger = {}
        params_logger['obj_radius'] = self.push_data_file['obj_radius']
        self.logger = Logger(params_logger)
        self.time_profile = False

        if cfg.options.random_seed is not None:
            np.random.seed(cfg.options.random_seed)

        fig1 = plt.figure(constrained_layout=True, figsize=(12, 8))

    def load_push_dataset(self):

        dataset_file = "{0}/{1}/{2}.json".format(
            BASE_PATH, self.cfg.dataio.srcdir_dataset, self.cfg.dataio.dataset_name)
        with open(dataset_file) as f:
            dataset = json.load(f)

        self.push_data_file['ee_poses_2d'] = np.array(dataset['ee_poses_2d'])
        self.push_data_file['obj_poses_2d'] = np.array(dataset['obj_poses_2d'])
        self.push_data_file['img_feats'] = np.array(dataset['img_feats'])

        self.push_data_file['contact_flag'] = np.array(dataset['contact_flag'])
        self.push_data_file['contact_episode'] = np.array(
            dataset['contact_episode'])

        params = dataset['params']
        self.push_data_file['obj_radius'] = params['obj_radius']

    def load_object_sdf(self):
        sdf_file = "{0}/{1}/{2}.json".format(
            BASE_PATH, self.cfg.dataio.srcdir_dataset, self.cfg.dataio.obj_sdf_shape)
        with open(sdf_file) as f:
            sdf_data = json.load(f)

        cell_size = sdf_data['grid_res']
        sdf_cols = sdf_data['grid_size_x']
        sdf_rows = sdf_data['grid_size_y']
        sdf_data_vec = sdf_data['grid_data']
        sdf_origin_x = sdf_data['grid_origin_x']
        sdf_origin_y = sdf_data['grid_origin_y']

        origin = gtsam.Point2(sdf_origin_x, sdf_origin_y)

        sdf_data_mat = np.zeros((sdf_rows, sdf_cols))
        for i in range(sdf_data_mat.shape[0]):
            for j in range(sdf_data_mat.shape[1]):
                sdf_data_mat[i, j] = sdf_data_vec[i][j]

        self.object_sdf = pushest.PlanarSDF(origin, cell_size, sdf_data_mat)

    def init_optimizer(self):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.setRelinearizeSkip(10)

        self.isam2 = gtsam.ISAM2(params)
        self.isam2_estimate = None

    def init_noise_models(self):

        self.first_pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.first_pose_prior))
        self.obj_pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.obj_pose_prior))
        self.ee_pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.ee_pose_prior))
        self.obj_pose_interseq_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.obj_pose_interseq_noise))

        self.odom_motion_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.odom_motion))
        self.qs_push_motion_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.qs_push_motion))
        self.sdf_intersection_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.sdf_intersection))

        self.tactile_rel_meas_noise = gtsam.noiseModel_Diagonal.Sigmas(
            np.array(self.cfg.noise_models.tactile_rel_meas))

        self.sampler_obj_pose_prior_noise = gtsam.Sampler(
            self.obj_pose_prior_noise, 0)
        self.sampler_ee_pose_prior_noise = gtsam.Sampler(
            self.ee_pose_prior_noise, 0)

    def optimizer_update(self):
        self.isam2.update(self.graph, self.init_vals)
        self.isam2_estimate = self.isam2.calculateEstimate()

    def reset_graph(self):
        self.graph.resize(0)
        self.init_vals.clear()

    def add_gaussian_noise(self, pose, noisevec, add_noise=True):
        if (add_noise):
            pose_noisy = pose.retract(noisevec)
        else:
            pose_noisy = pose

        return pose_noisy

    def normalize_imgfeat(self, img_feat):
        img_feat_norm = np.divide(np.subtract(
            img_feat, self.cfg.data_process.mean_img_feat), self.cfg.data_process.std_img_feat)

        return img_feat_norm

    def log_step(self, tstep):

        num_data = tstep + 1

        # log estimated poses
        pose_vals_graph = self.isam2.calculateEstimate()
        obj_poses2d_graph = np.zeros((num_data, 3))
        ee_poses2d_graph = np.zeros((num_data, 3))
        for i in range(0, num_data):
            obj_key = gtsam.symbol(ord('o'), i)
            ee_key = gtsam.symbol(ord('e'), i)

            obj_pose2d = pose_vals_graph.atPose2(obj_key)
            ee_pose2d = pose_vals_graph.atPose2(ee_key)

            obj_poses2d_graph[i, :] = [
                obj_pose2d.x(), obj_pose2d.y(), obj_pose2d.theta()]
            ee_poses2d_graph[i, :] = [
                ee_pose2d.x(), ee_pose2d.y(), ee_pose2d.theta()]
        self.logger.log_step("poses2d/obj_graph", obj_poses2d_graph, tstep)
        self.logger.log_step("poses2d/ee_graph", ee_poses2d_graph, tstep)

        # log gt poses
        obj_poses2d_gt = self.push_data['obj_poses_2d'][self.push_data_idxs[0:num_data]]
        ee_poses2d_gt = self.push_data['ee_poses_2d'][self.push_data_idxs[0:num_data]]
        self.logger.log_step('poses2d/obj_gt', obj_poses2d_gt, tstep)
        self.logger.log_step('poses2d/ee_gt', ee_poses2d_gt, tstep)

    def tactile_factor_error(self, factor, push_idx_i, push_idx_j, err_vec):
        pose_rel_meas = factor.getMeasTransform()
        obj_pose_gt_i = gtsam.Pose2(self.push_data['obj_poses_2d'][push_idx_i])
        obj_pose_gt_j = gtsam.Pose2(self.push_data['obj_poses_2d'][push_idx_j])
        ee_pose_gt_i = gtsam.Pose2(self.push_data['ee_poses_2d'][push_idx_i])
        ee_pose_gt_j = gtsam.Pose2(self.push_data['ee_poses_2d'][push_idx_j])
        pose_rel_gt = factor.getExpectTransform(
            obj_pose_gt_i, obj_pose_gt_j, ee_pose_gt_i, ee_pose_gt_j)
        err_pose = pose_rel_gt.between(pose_rel_meas)
        err_vec.append(gtsam.Pose2.Logmap(err_pose))

        # print("*******************\n")
        # print("pose_rel_gt: {0}\n pose_rel_meas: {1}".format(pose_rel_gt, pose_rel_meas))
        # print("err_pose: {0}\n err_vec: {1}".format(err_pose, gtsam.Pose2.Logmap(err_pose)))
        # print("*******************\n")

        return err_vec

    def print_step(self, tstep):
        obj_key = gtsam.symbol(ord('o'), tstep)
        ee_key = gtsam.symbol(ord('e'), tstep)
        print('Estimated current object pose: \n {0} '.format(
            self.isam2_estimate.atPose2(obj_key)))
        print('Groundtruth current object pose: \n {0} '.format(
            self.push_data['obj_poses_2d'][tstep]))
        print('Estimated current endeff pose: \n {0} '.format(
            self.isam2_estimate.atPose2(ee_key)))
        print('Groundtruth current endeff pose: \n {0} '.format(
            self.push_data['ee_poses_2d'][tstep]))

    def visualize_step(self, tstep):
        qviz.draw_poses_step(self.logger, tstep, 'red',
                             self.cfg.factors.prefix, self.cfg.dataio.obj_sdf_shape)

        # self.visualizer.plot_pose_errors_step(tstep, self.logger.poses[tstep]['obj_poses2d_gt'],
        #   self.logger.poses[tstep]['obj_poses2d_graph'], 'red', self.cfg.factors.prefix)

    def save_logger_obj(self, prefix=''):
        logger_dir = "{0}/local/outputs/{1}/seq_{2:03d}".format(
            BASE_PATH, self.cfg.dataio.dataset_name, self.seq_idx)
        dir_utils.make_dir(logger_dir)

        logger_filename = "{0}/{1}.obj".format(logger_dir, prefix)
        f = open(logger_filename, 'wb')
        pkl.dump(self.logger, f)
        f.close()

        print("Saved logger object for seq {0} to {1}".format(
            self.seq_idx, logger_filename))
    
    def init_obj_ee_vars(self, tstep, init_mode=0):
  
        obj_key_tm1 = gtsam.symbol(ord('o'), tstep - 1)
        obj_key_tm2 = gtsam.symbol(ord('o'), tstep - 2) if tstep > 1 else obj_key_tm1
        ee_key_tm1 = gtsam.symbol(ord('e'), tstep - 1)
        ee_key_tm2 = gtsam.symbol(ord('e'), tstep - 2) if tstep > 1 else ee_key_tm1

        if (init_mode == 0):      
            obj_key0_init = self.isam2_estimate.atPose2(obj_key_tm1)
            ee_key0_init = self.isam2_estimate.atPose2(ee_key_tm1)
        elif (init_mode == 1):
            obj_pose_delta = self.isam2_estimate.atPose2(obj_key_tm2).between(self.isam2_estimate.atPose2(obj_key_tm1))
            ee_pose_delta = self.isam2_estimate.atPose2(ee_key_tm2).between(self.isam2_estimate.atPose2(ee_key_tm1))
            obj_key0_init = self.isam2_estimate.atPose2(obj_key_tm1).compose(obj_pose_delta)
            ee_key0_init = self.isam2_estimate.atPose2(ee_key_tm1).compose(ee_pose_delta)
        else:
            print("Init mode {0} not defined".format(init_mode))
        
        return (obj_key0_init, ee_key0_init)

    def enable_tactile_oracle(self, factor, push_idx_i, push_idx_j):
        obj_pose_gt_i = utils.vec3_to_pose2(
            self.push_data['obj_poses_2d'][push_idx_i])
        obj_pose_gt_j = utils.vec3_to_pose2(
            self.push_data['obj_poses_2d'][push_idx_j])
        ee_pose_gt_i = utils.vec3_to_pose2(
            self.push_data['ee_poses_2d'][push_idx_i])
        ee_pose_gt_j = utils.vec3_to_pose2(
            self.push_data['ee_poses_2d'][push_idx_j])

        pose_rel_gt = factor.getExpectTransform(
            obj_pose_gt_i, obj_pose_gt_j, ee_pose_gt_i, ee_pose_gt_j)
        factor.setOracle(True, pose_rel_gt)

        return factor

    def add_prior_pose_factor(self, key, pose_meas):

        self.init_vals.insert(key, pose_meas)
        self.graph.push_back(gtsam.PriorFactorPose2(
            key, pose_meas, self.first_pose_prior_noise))

    def add_binary_pose_factor(self, key1, key2, pose_meas):
        self.graph.push_back(gtsam.BetweenFactorPose2(
            key1, key2, pose_meas, self.obj_pose_interseq_noise))

    def add_unary_obj_pose_factor(self, obj_key, pose_meas):
        self.graph.push_back(gtsam.PriorFactorPose2(
            obj_key, pose_meas, self.obj_pose_prior_noise))

    def add_unary_ee_pose_factor(self, ee_key, pose_meas):
        self.graph.push_back(gtsam.PriorFactorPose2(
            ee_key, pose_meas, self.ee_pose_prior_noise))

    def add_qs_motion_factor(self, obj_key0, obj_key1, ee_key0, ee_key1):
        if (self.cfg.dataio.obj_sdf_shape == 'disc'):
            c_sq = math.pow(self.push_data['obj_radius'] / 3, 2)
        elif (self.cfg.dataio.obj_sdf_shape == 'rect'):
            c_sq = math.pow(math.sqrt(0.2363**2 + 0.1579**2), 2)
        elif (self.cfg.dataio.obj_sdf_shape == 'ellip'):
            c_sq = (0.5*(0.1638 + 0.2428)) ** 2
        
        self.graph.push_back(pushest.QSVelPushMotionRealObjEEFactor(
            obj_key0, obj_key1, ee_key0, ee_key1, c_sq, self.qs_push_motion_noise))

    def add_sdf_intersection_factor(self, obj_key1, ee_key1):
        self.graph.push_back(pushest.IntersectionPlanarSDFObjEEFactor(
            obj_key1, ee_key1, self.object_sdf, self.ee_radius, self.sdf_intersection_noise))

    def add_tactile_rel_meas_factor(self, pose_idx, tstep):

        err_vec = []
        if (pose_idx-1 > self.cfg.factors.tactile_min_offset):
            max_offset = np.minimum(
                pose_idx-1, self.cfg.factors.tactile_max_offset)
            for offset in range(self.cfg.factors.tactile_min_offset, max_offset):

                pose_idx_i = pose_idx - 1 - offset
                pose_idx_j = pose_idx - 1

                if (self.episode_idxs[pose_idx_i] != self.episode_idxs[pose_idx_j]):
                    continue

                push_idx_i = self.push_data_idxs[pose_idx_i]
                push_idx_j = self.push_data_idxs[pose_idx_j]

                img_feat_i = self.push_data['img_feats'][push_idx_i]
                img_feat_j = self.push_data['img_feats'][push_idx_j]
                img_feat_i = self.normalize_imgfeat(
                    img_feat_i) if self.cfg.data_process.norm_img_feat else img_feat_i
                img_feat_j = self.normalize_imgfeat(
                    img_feat_j) if self.cfg.data_process.norm_img_feat else img_feat_j

                obj_key_i = gtsam.symbol(ord('o'), pose_idx_i)
                obj_key_j = gtsam.symbol(ord('o'), pose_idx_j)
                ee_key_i = gtsam.symbol(ord('e'), pose_idx_i)
                ee_key_j = gtsam.symbol(ord('e'), pose_idx_j)

                factor = pushest.TactileRelativeTfRegressionFactor(obj_key_i, obj_key_j, ee_key_i, ee_key_j, img_feat_i, img_feat_j,
                                                                   self.torch_model_file, self.tactile_rel_meas_noise)
                factor.setFlags(self.cfg.factors.yaw_only_error,
                                self.cfg.factors.constant_model)
                factor.setLabel(self.cfg.dataio.class_label,
                                self.cfg.dataio.num_classes)

                if self.cfg.factors.tactile_oracle:
                    factor = self.enable_tactile_oracle(
                        factor, push_idx_i, push_idx_j)

                self.graph.push_back(factor)

                err_vec = self.tactile_factor_error(
                    factor, push_idx_i, push_idx_j, err_vec)

        self.logger.log_step("errors/tactile_factor", err_vec, tstep)

    def set_factor_mode_config(self, mode):
        if (mode == 0):
            self.cfg.factors.prefix = "qs+sdf"
            self.cfg.factors.enable_tactile_rel_meas = False
            self.cfg.factors.constant_model = False
            self.cfg.factors.tactile_oracle = False

        elif (mode == 1):
            self.cfg.factors.prefix = "qs+sdf+tactile-linear"
            self.cfg.factors.enable_tactile_rel_meas = True
            self.cfg.factors.constant_model = False
            self.cfg.factors.tactile_oracle = False

        elif (mode == 2):
            self.cfg.factors.prefix = "qs+sdf+tactile-const"
            self.cfg.factors.enable_tactile_rel_meas = True
            self.cfg.factors.constant_model = True
            self.cfg.factors.tactile_oracle = False

        elif (mode == 3):
            self.cfg.factors.prefix = "qs+sdf+tactile-oracle"
            self.cfg.factors.enable_tactile_rel_meas = True
            self.cfg.factors.constant_model = False
            self.cfg.factors.tactile_oracle = True

    def run_episode(self):

        self.init_optimizer()
        self.reset_graph()

        num_steps = self.cfg.dataio.num_steps  # len(self.push_data_idxs)
        skip_pose = 1
        pose_idx, prev_pose_idx = -skip_pose, -skip_pose
        runtime = np.zeros((num_steps, 1))

        for tstep in range(num_steps):

            # print("Time step: {0}".format(tstep))

            prev_pose_idx = pose_idx
            pose_idx = pose_idx + skip_pose
            push_idx = self.push_data_idxs[pose_idx]

            if (tstep == 0):
                obj_pose_prior = utils.vec3_to_pose2(
                    self.push_data['obj_poses_2d'][push_idx])
                obj_key0 = gtsam.symbol(ord('o'), tstep)
                self.add_prior_pose_factor(obj_key0, obj_pose_prior)
                ee_key0 = gtsam.symbol(ord('e'), tstep)
                ee_pose_prior = self.push_data['ee_poses_2d'][push_idx]
                self.add_prior_pose_factor(
                    ee_key0, gtsam.Pose2(ee_pose_prior[0], ee_pose_prior[1], ee_pose_prior[2]))

                if self.time_profile:
                    tstart = time.time()

                self.optimizer_update()

                if self.time_profile:
                    runtime_step = time.time() - tstart
                    runtime[tstep] = runtime_step

                # log, print, visualize
                self.log_step(tstep)

                if (self.cfg.options.vis_step_flag == True):
                    # self.print_step(tstep)
                    self.visualize_step(tstep)

                self.reset_graph()

                continue
            
            obj_key0 = gtsam.symbol(ord('o'), tstep - 1)
            obj_key1 = gtsam.symbol(ord('o'), tstep)
            ee_key0 = gtsam.symbol(ord('e'), tstep - 1)
            ee_key1 = gtsam.symbol(ord('e'), tstep)

            # groundtruth poses from push dataset (noisy estimates used in ee, obj pose priors)       
            obj_pose_curr_gt = utils.vec3_to_pose2(self.push_data['obj_poses_2d'][push_idx])
            ee_pose_curr_gt = utils.vec3_to_pose2(self.push_data['ee_poses_2d'][push_idx])

            obj_key1_init, ee_key1_init = self.init_obj_ee_vars(tstep, init_mode=0)
            self.init_vals.insert(obj_key1, obj_key1_init)
            self.init_vals.insert(ee_key1, ee_key1_init)

            # add factor: interseq obj binary pose factor
            if (self.episode_idxs[prev_pose_idx] != self.episode_idxs[pose_idx]):
                self.add_binary_pose_factor(obj_key0, obj_key1, gtsam.Pose2(0.0, 0.0, 0.0))

            # add factor: ee unary pose factor
            ee_pose_curr_gt_noisy = self.add_gaussian_noise(
                ee_pose_curr_gt, self.sampler_ee_pose_prior_noise.sample())
            self.add_unary_ee_pose_factor(ee_key1, ee_pose_curr_gt_noisy)

            # add factor: obj unary pose factor (periodic)
            if (pose_idx % self.cfg.factors.obj_prior_interval == 0):
                obj_pose_curr_gt_noisy = self.add_gaussian_noise(
                    obj_pose_curr_gt, self.sampler_obj_pose_prior_noise.sample())
                self.add_unary_obj_pose_factor(
                    obj_key1, obj_pose_curr_gt_noisy)

             # add factor: quasi-static motion factor
            if ((self.push_data['contact_flag'][push_idx]) & (self.cfg.factors.enable_qs_motion)):
                if (self.episode_idxs[prev_pose_idx] == self.episode_idxs[pose_idx]):
                    self.add_qs_motion_factor(
                        obj_key0, obj_key1, ee_key0, ee_key1)

             # add factor: sdf intersection factor
            if ((self.push_data['contact_flag'][push_idx]) & (self.cfg.factors.enable_sdf_intersection)):
                self.add_sdf_intersection_factor(obj_key1, ee_key1)

             # add factor: tactile relative meas factor
            if ((self.push_data['contact_flag'][push_idx]) & (self.cfg.factors.enable_tactile_rel_meas)):
                self.add_tactile_rel_meas_factor(pose_idx, tstep)

            self.optimizer_update()

            if self.time_profile:
                runtime_step = time.time() - tstart
                runtime[tstep] = runtime_step

            # log, print, visualize
            self.log_step(tstep)

            if (self.cfg.options.vis_step_flag == True):
                # self.print_step(tstep)
                self.visualize_step(tstep)

            self.reset_graph()

        self.visualize_step(num_steps-1)
        self.logger.log_param('num_steps', num_steps)
        self.logger.log_param('episode_idxs', self.episode_seq)

        if self.time_profile:
            self.logger.log_runtime(runtime)

        self.save_logger_obj(self.cfg.factors.prefix)

    def sequence_generator(self):

        episode_idxs = list(set(self.push_data_file['contact_episode']))
        episode_seq_list = list(permutations(episode_idxs, self.cfg.dataio.num_eps_seq))

        random.shuffle(episode_seq_list)

        # episode_seq_list = [(23, 25, 26)] # disc qual result [trial1 dataset]
        # episode_seq_list = [(1, 6, 8)] # rect qual result [corners dataset]
        # episode_seq_list = [(1, 5, 3)] # ellip

        print("[RealPushGraphOptimizeISAM2::sequence_generator] Generated {0} episode sequences".format(len(episode_seq_list)))

        return episode_seq_list

    def run(self):

        episode_seq_list = self.sequence_generator()

        self.seq_idx = 0
        num_seqs = np.minimum(len(episode_seq_list), self.cfg.dataio.num_seqs)
        for episode_seq in tqdm(episode_seq_list):

            self.push_data_idxs = []
            self.episode_idxs = []
            self.episode_seq = list(episode_seq)
            for episode in self.episode_seq:
                push_data_idxs_curr = [idx for idx, val in enumerate(
                    self.push_data_file['contact_episode']) if (val == episode)]
                self.push_data_idxs.append(push_data_idxs_curr)
                self.episode_idxs.append([episode] * len(push_data_idxs_curr))

            self.push_data_idxs = [
                item for sublist in self.push_data_idxs for item in sublist]
            self.episode_idxs = [
                item for sublist in self.episode_idxs for item in sublist]

            if (len(self.episode_idxs) < self.cfg.dataio.num_steps):
                continue

            print("Running episode sequence {0}/{1} of length {2} and idxs {3}".format(
                self.seq_idx, num_seqs, len(self.episode_idxs), self.episode_seq))
            self.push_data = data_process.transform_episodes_common_frame(
                self.episode_seq, self.push_data_file)
            
            for factor_mode in [0, 1, 2, 3]:
                try:
                    self.set_factor_mode_config(factor_mode)
                    self.run_episode()
                except KeyboardInterrupt:
                    sys.exit(0)
                except:
                    print(sys.exc_info()[0])
                    self.seq_idx = self.seq_idx - 1
                    break

            self.seq_idx = self.seq_idx + 1
            if (self.seq_idx > num_seqs):
                break            

        print("Saved results for {0} sequences of {1} dataset.".format(self.seq_idx, self.cfg.dataio.dataset_name))

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(BASE_PATH, "python/config/real_push_graph.yaml")


@hydra.main(config_path=config_path, strict=False)
def main(cfg):
    real_push = RealPushGraphOptimizeISAM2(cfg)
    real_push.run()


if __name__ == '__main__':
    main()
