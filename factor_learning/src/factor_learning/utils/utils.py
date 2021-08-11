
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import pytorch3d.transforms as p3d_t

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def tf2d_between(pose1, pose2, device=None):
    """
    Relative transform of pose2 in pose1 frame, i.e. T12 = T1^{1}*T2
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose2: n x 3 tensor [x,y,yaw]
    :return: pose12 n x 3 tensor [x,y,yaw]
    """

    num_data = pose1.shape[0]

    rot1 = torch.cat([torch.zeros(num_data, 1, device=device), torch.zeros(
        num_data, 1, device=device), pose1[:, 2][:, None]], 1)
    rot2 = torch.cat([torch.zeros(num_data, 1, device=device), torch.zeros(
        num_data, 1, device=device), pose2[:, 2][:, None]], 1)
    t1 = torch.cat([pose1[:, 0][:, None], pose1[:, 1]
                    [:, None], torch.zeros(num_data, 1, device=device)], 1)
    t2 = torch.cat([pose2[:, 0][:, None], pose2[:, 1]
                    [:, None], torch.zeros(num_data, 1, device=device)], 1)

    R1 = p3d_t.euler_angles_to_matrix(rot1, "XYZ")
    R2 = p3d_t.euler_angles_to_matrix(rot2, "XYZ")
    R1t = torch.inverse(R1)

    R12 = torch.matmul(R1t, R2)
    rot12 = p3d_t.matrix_to_euler_angles(R12, "XYZ")
    t12 = torch.matmul(R1t, (t2-t1)[:, :, None])
    t12 = t12[:, :, 0]

    tx = t12[:, 0][:, None]
    ty = t12[:, 1][:, None]
    yaw = rot12[:, 2][:, None]
    pose12 = torch.cat([tx, ty, yaw], 1)

    return pose12

def tf2d_compose(pose1, pose12):
    """
    Composing pose1 with pose12, i.e. T2 = T1*T12
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose12: n x 3 tensor [x,y,yaw]
    :return: pose2 n x 3 tensor [x,y,yaw]
    """

    num_data = pose1.shape[0]

    rot1 = torch.cat([torch.zeros(num_data, 1), torch.zeros(
        num_data, 1), pose1[:, 2][:, None]], 1)
    rot12 = torch.cat([torch.zeros(num_data, 1), torch.zeros(
        num_data, 1), pose12[:, 2][:, None]], 1)
    t1 = torch.cat([pose1[:, 0][:, None], pose1[:, 1]
                    [:, None], torch.zeros(num_data, 1)], 1)
    t12 = torch.cat([pose12[:, 0][:, None], pose12[:, 1]
                    [:, None], torch.zeros(num_data, 1)], 1)

    R1 = p3d_t.euler_angles_to_matrix(rot1, "XYZ")
    R12 = p3d_t.euler_angles_to_matrix(rot12, "XYZ")

    R2 = torch.matmul(R1, R12)
    rot2 = p3d_t.matrix_to_euler_angles(R2, "XYZ")
    t2 = torch.matmul(R1, t12[:, :, None]) + t1[:, :, None]
    t2 = t2[:, :, 0]

    tx = t2[:, 0][:, None]
    ty = t2[:, 1][:, None]
    yaw = rot2[:, 2][:, None]
    pose2 = torch.cat([tx, ty, yaw], 1)

    return pose2


def tf2d_net_input(pose_rel):
    tx = pose_rel[:, 0][:, None]  # N x 1
    ty = pose_rel[:, 1][:, None]  # N x 1
    yaw = pose_rel[:, 2][:, None]  # N x 1
    pose_rel_net = torch.cat(
        [tx*1000, ty*1000, torch.cos(yaw), torch.sin(yaw)], 1)  # N x 4
    return pose_rel_net


def regularization_loss(params_net, reg_type):
    reg_loss = 0
        
    if(reg_type == "l1"):
        for param in params_net:
            reg_loss += torch.sum(torch.abs(param))
    
    elif(reg_type == "l2"):
        for param in params_net:
            reg_loss += torch.sum(torch.norm(param))
    
    return reg_loss

def network_update(output, optimizer):
    # clear, backprop and apply new gradients
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

def normalize_imgfeats(imgfeat, norm_mean_list, norm_std_list, device=None):
    norm_mean = torch.cuda.FloatTensor(norm_mean_list) if device == torch.device(
        'cuda:0') else torch.FloatTensor(norm_mean_list)
    norm_std = torch.cuda.FloatTensor(norm_std_list) if device == torch.device(
        'cuda:0') else torch.FloatTensor(norm_std_list)

    imgfeat_norm = torch.div(torch.sub(imgfeat, norm_mean), norm_std)
    
    return imgfeat_norm

def denormalize_img(img_norm, norm_mean, norm_std):
    img = torch.zeros(img_norm.shape)

    img[:, 0, :, :] = torch.add(
        torch.mul(img_norm[:, 0, :, :], norm_std[0]), norm_mean[0])
    img[:, 1, :, :] = torch.add(
        torch.mul(img_norm[:, 1, :, :], norm_std[1]), norm_mean[1])
    img[:, 2, :, :] = torch.add(
        torch.mul(img_norm[:, 2, :, :], norm_std[2]), norm_mean[2])

    return img

def weighted_mse_loss(input, target, mse_wts):
    return torch.mean(mse_wts * (input - target) ** 2)

def make_dir(dir, clear_dir=False, print_status=False):
    if print_status:
        print("Creating directory {0}".format(dir))

    cmd = "mkdir -p {0}".format(dir)
    os.popen(cmd, 'r')

    if clear_dir:
        cmd = "rm -rf {0}/*".format(dir)
        os.popen(cmd, 'r')

def load_pkl_obj(filename):
    with (open(filename, "rb")) as f:
        pkl_obj = pkl.load(f)
    f.close()

    return pkl_obj