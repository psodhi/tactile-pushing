#!/usr/bin/env python

import math
import numpy as np
from tqdm import tqdm

import seaborn as sns
import pandas as pd

from subprocess import call
import matplotlib.pyplot as plt

from pushestpy.utils import utils

plt.rcParams.update({'font.size': 36})


def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap


def traj_error(xyh_gt, xyh_est, err_type):

    if (err_type == "rmse"):
        diff = xyh_gt - xyh_est
        diff[:, 2] = wrap_to_pi(diff[:, 2])
        diff_sq = diff**2

        rmse_trans = np.sqrt(np.mean(diff_sq[:, 0:2].flatten()))
        rmse_rot = np.sqrt(np.mean(diff_sq[:, 2].flatten()))

        error = (rmse_trans, rmse_rot)

    elif (err_type == "ate"):
        pass

    return error


def compute_traj_errors(cfg):

    errors_all = {}

    for logger_idx, logger_name in enumerate(cfg.logger.names):
        print("Computing errors for logger {0}".format(logger_name))
        error_mat = np.zeros((cfg.logger.num_seqs, cfg.logger.num_steps, 2))
        for seq_idx in tqdm(range(cfg.logger.num_seqs)):
            logger_filename = "{0}/local/outputs/{1}/seq_{2:03d}/{3}.obj".format(
                cfg.BASE_PATH, cfg.dataset_names[0], seq_idx, logger_name)
            logger = utils.load_pkl_obj(logger_filename)

            for tstep in range(cfg.logger.num_steps):
                poses_obj_gt = logger.data[tstep][cfg.logger.fields.poses_obj_gt]
                poses_obj_graph = logger.data[tstep][cfg.logger.fields.poses_obj_graph]

                error_mat[seq_idx, tstep, 0], error_mat[seq_idx, tstep, 1] = traj_error(
                    poses_obj_gt, poses_obj_graph, err_type="rmse")

        errors_all[logger_name] = error_mat

    return errors_all


def compute_traj_errors_multi_dataset(cfg):

    errors_all = {}
    num_seqs_total = sum(cfg.logger.num_seqs)

    for logger_idx, logger_name in enumerate(cfg.logger.names):
        print("Computing errors for logger {0}".format(logger_name))

        error_mat = np.zeros((num_seqs_total, cfg.logger.num_steps, 2))
        err_idx = 0

        for (ds_idx, dataset) in enumerate(cfg.dataset_names):
            for seq_idx in tqdm(range(cfg.logger.num_seqs[ds_idx])):

                logger_filename = "{0}/local/outputs/{1}/{2}/seq_{3:03d}/{4}.obj".format(
                    cfg.BASE_PATH, cfg.logger.dir_prefix, cfg.dataset_names[ds_idx], seq_idx, logger_name)
                logger = utils.load_pkl_obj(logger_filename)

                for tstep in range(cfg.logger.num_steps):
                    poses_obj_gt = logger.data[tstep][cfg.logger.fields.poses_obj_gt]
                    poses_obj_graph = logger.data[tstep][cfg.logger.fields.poses_obj_graph]

                    error_mat[err_idx, tstep, 0], error_mat[err_idx, tstep, 1] = traj_error(
                        poses_obj_gt, poses_obj_graph, err_type="rmse")

                err_idx = err_idx + 1

        errors_all[logger_name] = error_mat

    return errors_all


def get_subset(errors, cfg, num_subset):

    # remove outliers
    errors_logger = errors[cfg.logger.names[2]]  # nseq x nsteps x 2
    [nseq, nsteps, ndim] = errors_logger.shape
    nout = np.minimum(int(0.25 * nseq), nseq-num_subset)
    outlier_idxs = np.argsort(-errors_logger[:, -1, 1])[:nout]
    for logger_name in cfg.logger.names:
        errors[logger_name] = np.delete(errors[logger_name], outlier_idxs, 0)

    # choose random subset
    subset_idxs = np.random.randint(
        len(errors[cfg.logger.names[0]]), size=num_subset)
    for logger_name in cfg.logger.names:
        errors[logger_name] = errors[logger_name][subset_idxs][0:num_subset, :, :]

    return errors


def plot_traj_errors(cfg, errors):

    plt.ion()
    fig1 = plt.figure(constrained_layout=True, figsize=(12, 8))
    fig2 = plt.figure(constrained_layout=True, figsize=(12, 8))
    fig3 = plt.figure(constrained_layout=True, figsize=(12, 8))
    fig4 = plt.figure(constrained_layout=True, figsize=(12, 8))
    # fontsize_label = 22

    num_subset = 50
    errors = get_subset(errors, cfg, num_subset=num_subset) if (
        num_subset < sum(cfg.logger.num_seqs)) else errors
    num_seqs = np.minimum(num_subset, sum(cfg.logger.num_seqs))

    errors_final_bplot = np.zeros(
        (num_seqs, len(cfg.logger.names), 2))  # nseq x nloggers x 2
    for logger_idx, logger_name in enumerate(cfg.logger.names):

        errors_logger = errors[logger_name]  # nseq x nsteps x 2
        errors_final_bplot[:, logger_idx, 0] = 1000 * errors_logger[:, -1, 0]
        errors_final_bplot[:, logger_idx, 1] = 180 / \
            math.pi * (errors_logger[:, -1, 1])

        # plotting data
        x = np.arange(cfg.logger.num_steps) / float(cfg.logger.freq)
        mean_error_time = np.mean(errors_logger, 0)
        std_error_time = np.std(errors_logger, 0)
        scale_vis_std = 0.3

        # convert units
        mean_error_time[:, 0] = 1000 * mean_error_time[:, 0]
        std_error_time[:, 0] = 1000 * std_error_time[:, 0]
        mean_error_time[:, 1] = 180/math.pi * (mean_error_time[:, 1])
        std_error_time[:, 1] = 180/math.pi * (std_error_time[:, 1])

        # error shade plot (trans)
        plt.figure(fig1.number)
        plt.semilogy(x, mean_error_time[:, 0], color=cfg.logger.colors[logger_idx],
                     label=cfg.logger.labels[logger_idx], linewidth=2)
        plt.fill_between(x, mean_error_time[:, 0] - scale_vis_std * std_error_time[:, 0],
                         mean_error_time[:, 0] +
                         scale_vis_std * std_error_time[:, 0],
                         color=cfg.logger.colors[logger_idx], alpha=0.2)
        plt.xlabel("Time (s)")
        plt.ylabel("RMSE translational error (mm)")
        # plt.legend(loc='upper left')
        plt.ylim((10**-1, 10**2))
        plt.savefig("{0}/local/visualizations/quant_{1}_trans_err_shadeplot.png".format(
            cfg.BASE_PATH, cfg.obj_sdf_shape))

        # error shade plot (rot)
        plt.figure(fig2.number)
        plt.semilogy(x, mean_error_time[:, 1], color=cfg.logger.colors[logger_idx],
                     label=cfg.logger.labels[logger_idx], linewidth=2)
        plt.fill_between(x, mean_error_time[:, 1] - scale_vis_std * std_error_time[:, 1],
                         mean_error_time[:, 1] +
                         scale_vis_std * std_error_time[:, 1],
                         color=cfg.logger.colors[logger_idx], alpha=0.2)
        plt.xlabel("Time (s)")
        plt.ylabel("RMSE rotational error (deg)")
        # plt.legend(loc='upper left')
        plt.ylim((10**-2, 10**2))
        plt.savefig("{0}/local/visualizations/quant_{1}_rot_err_shadeplot.png".format(
            cfg.BASE_PATH, cfg.obj_sdf_shape))

        plt.show()

    line_props = dict(color="black", alpha=1.0, linewidth=2)
    kwargs = {'vert': True, 'notch': False, 'patch_artist': True,
              'medianprops': line_props, 'whiskerprops': line_props}

    # error box plot (trans)
    plt.figure(fig3.number)
    bplot1 = plt.gca().boxplot(
        errors_final_bplot[:, :, 0], widths=0.4, labels=cfg.logger.labels, **kwargs)
    plt.gca().set_yscale('log')
    # plt.title("RMSE translational error (mm)")
    plt.ylim((10**-1, 10**2))

    for patch, color in zip(bplot1['boxes'], cfg.logger.colors):
        color = np.array([color[0], color[1], color[2], 0.5])
        patch.set_facecolor(color)
        patch.set_linewidth(2)
    plt.savefig("{0}/local/visualizations/quant_{1}_trans_err_boxplot.png".format(
        cfg.BASE_PATH, cfg.obj_sdf_shape))

    # error box plot (rot)
    plt.figure(fig4.number)
    bplot2 = plt.gca().boxplot(
        errors_final_bplot[:, :, 1], widths=0.4, labels=cfg.logger.labels, **kwargs)
    plt.gca().set_yscale('log')
    # plt.title("RMSE rotational error (deg)")
    plt.ylim((10**-2, 10**2))

    for patch, color in zip(bplot2['boxes'], cfg.logger.colors):
        color = np.array([color[0], color[1], color[2], 0.5])
        patch.set_facecolor(color)
        patch.set_linewidth(2)
    plt.savefig("{0}/local/visualizations/quant_{1}_rot_err_boxplot.png".format(
        cfg.BASE_PATH, cfg.obj_sdf_shape))

    plt.show(block=True)


def eval_traj_error(cfg):
    errors = compute_traj_errors_multi_dataset(cfg)
    plot_traj_errors(cfg, errors)


def runtime_plot(cfg):

    # learnt model
    logger_idx = 2
    logger_name = cfg.logger.names[logger_idx]
    num_seqs = cfg.logger.num_seqs[0]

    runtime_mat = np.zeros((num_seqs, cfg.logger.num_steps))  # nseq x nsteps
    for seq_idx in tqdm(range(num_seqs)):
        logger_filename = "{0}/local/outputs/runtime/{1}/seq_{2:03d}/{3}.obj".format(
            cfg.BASE_PATH, cfg.dataset_names[0], seq_idx, logger_name)
        logger = utils.load_pkl_obj(logger_filename)

        for tstep in range(cfg.logger.num_steps):
            runtime_mat[seq_idx, tstep] = logger.runtime[tstep]

    x = np.arange(cfg.logger.num_steps) / float(cfg.logger.freq)
    mean_runtimes = np.mean(runtime_mat, 0)
    std_runtimes = np.std(runtime_mat, 0)

    plt.errorbar(x, mean_runtimes, std_runtimes, linewidth=2,
                 color=cfg.logger.colors[logger_idx])

    plt.ylim(0, 0.05)
    plt.xlabel("Time (s)", fontsize=28)
    plt.ylabel("Runtime per iteration", fontsize=28)

    plt.show(block=True)
