#!/usr/bin/env python

import math
import numpy as np
from tqdm import tqdm

from pushestpy.utils import utils

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rcParams.update({'font.size': 16})


def draw_endeff(poses_ee, color="dimgray", label=None, ax=None):

    # plot contact point and normal
    plt.plot(poses_ee[-1][0], poses_ee[-1][1],
             'k*') if ax is None else ax.plot(poses_ee[-1][0], poses_ee[-1][1], 'k*')
    ori = poses_ee[-1, 2]
    sz_arw = 0.03
    (dx, dy) = (sz_arw * -math.sin(ori), sz_arw * math.cos(ori))
    plt.arrow(poses_ee[-1, 0], poses_ee[-1, 1], dx, dy, linewidth=2,
              head_width=0.001, color=color, head_length=0.01, fc='dimgray', ec='dimgray') if ax is None else ax.arrow(poses_ee[-1, 0], poses_ee[-1, 1], dx, dy, linewidth=2,
                                                                                                                       head_width=0.001, color=color, head_length=0.01, fc='dimgray', ec='dimgray')

    ee_radius = 0.0075
    circle = mpatches.Circle(
        (poses_ee[-1][0], poses_ee[-1][1]), color='dimgray', radius=ee_radius)
    plt.gca().add_patch(circle) if ax is None else ax.add_patch(circle)


def draw_object(poses_obj, shape="disc", color="dimgray", label=None, ax=None):

    linestyle_gt = '--' if (color == "dimgray") else '-'
    plt.plot(poses_obj[:, 0], poses_obj[:, 1], color=color,
             linestyle=linestyle_gt, label=label, linewidth=2, alpha=0.9) if ax is None else ax.plot(poses_obj[:, 0], poses_obj[:, 1], color=color,
                                                                                                     linestyle=linestyle_gt, label=label, linewidth=2, alpha=0.9)

    if (shape == "disc"):
        disc_radius = 0.088
        circ_obj = mpatches.Circle((poses_obj[-1][0], poses_obj[-1][1]), disc_radius,
                                   facecolor='None', edgecolor=color, linestyle=linestyle_gt, linewidth=2, alpha=0.9)
        plt.gca().add_patch(circ_obj) if ax is None else ax.add_patch(circ_obj)

        # cross-bars
        (x0, y0, yaw) = (poses_obj[-1][0],
                         poses_obj[-1][1], poses_obj[-1][2])
        r = disc_radius
        plt.plot([x0 + r * math.cos(yaw), x0 - r * math.cos(yaw)],
                 [y0 + r * math.sin(yaw), y0 - r * math.sin(yaw)],
                 linestyle=linestyle_gt, color=color, alpha=0.4) if ax is None else ax.plot([x0 + r * math.cos(yaw), x0 - r * math.cos(yaw)],
                                                                                            [y0 + r * math.sin(
                                                                                                yaw), y0 - r * math.sin(yaw)],
                                                                                            linestyle=linestyle_gt, color=color, alpha=0.4)
        plt.plot([x0 - r * math.sin(yaw), x0 + r * math.sin(yaw)],
                 [y0 + r * math.cos(yaw), y0 - r * math.cos(yaw)],
                 linestyle=linestyle_gt, color=color, alpha=0.4) if ax is None else ax.plot([x0 - r * math.sin(yaw), x0 + r * math.sin(yaw)],
                                                                                            [y0 + r * math.cos(
                                                                                                yaw), y0 - r * math.cos(yaw)],
                                                                                            linestyle=linestyle_gt, color=color, alpha=0.4)

    elif (shape == "rect"):
        rect_len_x = 0.2363
        rect_len_y = 0.1579

        yaw = poses_obj[-1][2]
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
        offset = np.matmul(R, np.array(
            [[0.5*rect_len_x], [0.5*rect_len_y]]))
        xb = poses_obj[-1][0] - offset[0]
        yb = poses_obj[-1][1] - offset[1]
        rect = mpatches.Rectangle((xb, yb), rect_len_x, rect_len_y, angle=(
            np.rad2deg(yaw)), facecolor='None', edgecolor=color, linestyle=linestyle_gt, linewidth=2)
        plt.gca().add_patch(rect) if ax is None else ax.add_patch(rect)

    elif (shape == "ellip"):
        ellip_len_x = 0.1638
        ellip_len_y = 0.2428

        xb = poses_obj[-1][0]
        yb = poses_obj[-1][1]
        yaw = poses_obj[-1][2]
        ellip = mpatches.Ellipse((xb, yb), ellip_len_x, ellip_len_y, angle=(
            np.rad2deg(yaw)), facecolor='None', edgecolor=color, linestyle=linestyle_gt, linewidth=2)
        plt.gca().add_patch(ellip) if ax is None else ax.add_patch(ellip)


def draw_poses_step(logger, tstep, color, label, obj_shape):
    plt.cla()
    plt.gca().axis('equal')
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))

    poses_obj_gt = logger.data[tstep]["poses2d/obj_gt"]
    poses_ee_gt = logger.data[tstep]["poses2d/ee_gt"]
    draw_endeff(poses_ee_gt, color="dimgray")
    draw_object(poses_obj_gt, shape=obj_shape,
                color="dimgray", label="groundtruth")

    poses_obj_graph = logger.data[tstep]["poses2d/obj_graph"]
    poses_ee_graph = logger.data[tstep]["poses2d/ee_graph"]

    draw_endeff(poses_ee_graph, color=color)
    draw_object(poses_obj_graph, shape=obj_shape, color=color, label=label)

    plt.legend(loc='upper left')
    plt.draw()
    plt.pause(1e-16)


def draw_poses_file_single(cfg):

    num_seqs = cfg.logger.num_seqs[0]
    dataset_name = cfg.dataset_names[0]

    for seq_idx in tqdm(range(cfg.logger.start_seq_idx, num_seqs)):

        plt.ion()
        plt.figure(constrained_layout=True, figsize=(12, 8))

        # num_steps = len(logger.data)
        for tstep in range(cfg.logger.num_steps-1, cfg.logger.num_steps):
            plt.cla()
            plt.gca().axis('equal')

            plt.xlim((-0.5, 1.0))
            plt.ylim((-0.5, 1.0))

            for logger_idx, logger_name in enumerate(cfg.logger.names):

                logger_filename = "{0}/local/outputs/{1}/seq_{2:03d}/{3}.obj".format(
                    cfg.BASE_PATH, dataset_name, seq_idx, logger_name)
                logger = utils.load_pkl_obj(logger_filename)
                if (logger_idx == 0):
                    poses_obj_gt = logger.data[tstep][cfg.logger.fields.poses_obj_gt]
                    poses_ee_gt = logger.data[tstep][cfg.logger.fields.poses_ee_gt]
                    draw_endeff(poses_ee_gt, color="dimgray")
                    draw_object(poses_obj_gt, shape=cfg.obj_sdf_shape,
                                color="dimgray", label="groundtruth")

                poses_obj_graph = logger.data[tstep][cfg.logger.fields.poses_obj_graph]
                poses_ee_graph = logger.data[tstep][cfg.logger.fields.poses_ee_graph]

                draw_endeff(poses_ee_graph,
                            color=cfg.logger.colors[logger_idx])
                draw_object(poses_obj_graph, shape=cfg.obj_sdf_shape,
                            color=cfg.logger.colors[logger_idx], label=cfg.logger.labels[logger_idx])

            plt.legend(loc='bottom left')
            plt.draw()
            plt.pause(1e-12)
            import pdb
            pdb.set_trace()


def draw_poses_file_collage(cfg):

    plt.ion()
    fig = plt.figure(constrained_layout=True, figsize=(16, 4))
    nrows = 1
    ncols = 4
    gs = GridSpec(nrows, ncols, figure=fig)
    ax = [None] * (nrows*ncols)

    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[0, 1])
    ax[2] = fig.add_subplot(gs[0, 2])
    ax[3] = fig.add_subplot(gs[0, 3])

    for (ds_idx, dataset_name) in enumerate(cfg.dataset_names):
        num_seqs = cfg.logger.num_seqs[ds_idx]
        for seq_idx in tqdm(range(cfg.logger.start_seq_idx, num_seqs)):

            for tstep in range(cfg.logger.num_steps-1, cfg.logger.num_steps):
                for a in ax:
                    a.cla()

                for logger_idx, logger_name in enumerate(cfg.logger.names):

                    logger_filename = "{0}/local/outputs/{1}/{2}/seq_{3:03d}/{4}.obj".format(
                        cfg.BASE_PATH, cfg.logger.dir_prefix, dataset_name, seq_idx, logger_name)
                    logger = utils.load_pkl_obj(logger_filename)

                    # groundtruth
                    poses_obj_gt = logger.data[tstep][cfg.logger.fields.poses_obj_gt]
                    poses_ee_gt = logger.data[tstep][cfg.logger.fields.poses_ee_gt]
                    draw_endeff(poses_ee_gt, color="dimgray",
                                ax=ax[logger_idx])
                    draw_object(poses_obj_gt, shape=cfg.obj_sdf_shape,
                                color="dimgray", label="groundtruth", ax=ax[logger_idx])

                    # graph poses
                    poses_obj_graph = logger.data[tstep][cfg.logger.fields.poses_obj_graph]
                    poses_ee_graph = logger.data[tstep][cfg.logger.fields.poses_ee_graph]
                    draw_endeff(poses_ee_graph,
                                color=cfg.logger.colors[logger_idx], ax=ax[logger_idx])
                    draw_object(poses_obj_graph, shape=cfg.obj_sdf_shape,
                                color=cfg.logger.colors[logger_idx], label=cfg.logger.labels[logger_idx], ax=ax[logger_idx])

                    ax[logger_idx].set_xlim((-0.5, 1.0))
                    ax[logger_idx].set_ylim((-0.5, 1.0))
                    ax[logger_idx].axis('equal')
                    ax[logger_idx].set_title(cfg.logger.labels[logger_idx])

                plt.draw()
                plt.pause(1e-12)

                save_fig = True
                if save_fig:
                    dirfig = "{0}/local/outputs/{1}/{2}/images".format(
                        cfg.BASE_PATH, cfg.logger.dir_prefix, cfg.dataset_names[ds_idx], seq_idx)
                    utils.make_dir(dirfig)
                    plt.savefig(
                        "{0}/seq{1:03d}_tstep{2:03d}.png".format(dirfig, seq_idx, tstep))

def draw_poses_file_video(cfg):

    plt.ion()
    save_fig = True
    figlist = []
    figlist = [plt.figure(constrained_layout=True, figsize=(12, 12))] * 4

    subdirs = ["no_tactile", "const_tactile", "learnt_tactile", "oracle_tactile"]

    for (ds_idx, dataset_name) in enumerate(cfg.dataset_names):
        num_seqs = cfg.logger.num_seqs[ds_idx]
        for seq_idx in tqdm(range(cfg.logger.start_seq_idx, num_seqs)):

            if save_fig:
                dirfig = "{0}/local/outputs/{1}/{2}/video/seq_{3:03d}".format(
                    cfg.BASE_PATH, cfg.logger.dir_prefix, cfg.dataset_names[ds_idx], seq_idx)
                utils.make_dir(dirfig)

                for subdir in subdirs:
                    utils.make_dir("{0}/{1}".format(dirfig, subdir))
            
            for tstep in tqdm(range(0, cfg.logger.num_steps)):

                for logger_idx, logger_name in enumerate(cfg.logger.names):

                    for fig in figlist:
                        plt.figure(fig.number)
                        plt.clf()
                        plt.gca().axis('equal')
                        plt.gca().axis('off')
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])

                    logger_filename = "{0}/local/outputs/{1}/{2}/seq_{3:03d}/{4}.obj".format(
                        cfg.BASE_PATH, cfg.logger.dir_prefix, dataset_name, seq_idx, logger_name)
                    logger = utils.load_pkl_obj(logger_filename)

                    xmin = np.amin(logger.data[cfg.logger.num_steps-1][cfg.logger.fields.poses_obj_gt], 0)[0] - 0.5
                    xmax = np.amax(logger.data[cfg.logger.num_steps-1][cfg.logger.fields.poses_obj_gt], 0)[0] + 0.5
                    ymin = np.amin(logger.data[cfg.logger.num_steps-1][cfg.logger.fields.poses_obj_gt], 0)[1] - 0.5
                    ymax = np.amax(logger.data[cfg.logger.num_steps-1][cfg.logger.fields.poses_obj_gt], 0)[1] + 0.5

                    # groundtruth
                    plt.figure(figlist[logger_idx].number)
                    poses_obj_gt = logger.data[tstep][cfg.logger.fields.poses_obj_gt]
                    poses_ee_gt = logger.data[tstep][cfg.logger.fields.poses_ee_gt]
                    draw_endeff(poses_ee_gt, color="dimgray", ax=None)
                    draw_object(poses_obj_gt, shape=cfg.obj_sdf_shape,
                                color="dimgray", label="groundtruth", ax=None)

                    # graph poses
                    poses_obj_graph = logger.data[tstep][cfg.logger.fields.poses_obj_graph]
                    poses_ee_graph = logger.data[tstep][cfg.logger.fields.poses_ee_graph]
                    draw_endeff(poses_ee_graph,
                                color=cfg.logger.colors[logger_idx], ax=None)
                    draw_object(poses_obj_graph, shape=cfg.obj_sdf_shape,
                                color=cfg.logger.colors[logger_idx], label=cfg.logger.labels[logger_idx], ax=None)
                    
                    plt.xlim((xmin, xmax))
                    plt.ylim((ymin, ymax))

                    # plt.show()
                    # plt.pause(1e-12)

                    if save_fig:
                        plt.savefig("{0}/{1}/tstep_{2:03d}.png".format(dirfig, subdirs[logger_idx], tstep))
            
            for subdir in subdirs:
                imgsrcdir = "{0}/{1}".format(dirfig, subdir)
                viddstfile = "{0}/seq_{1:03d}_{2}.mp4".format(dirfig, seq_idx, subdir)
                utils.write_video_ffmpeg(imgsrcdir, viddstfile)

def draw_poses_from_file(cfg):

    # draw_poses_file_single(cfg)
    # draw_poses_file_collage(cfg)
    draw_poses_file_video(cfg)