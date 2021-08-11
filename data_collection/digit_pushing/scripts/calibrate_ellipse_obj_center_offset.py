#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

from subprocess import call

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import math
import json
import glob
import os

plt.rcParams.update({'font.size': 18})
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def draw_ellip_with_points(x, y, x0, y0, a, b, th):

    plt.ion()
    plt.scatter(x, y, color="red", label="ellip-pts")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.gca().axis('equal')

    plt.show()

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    ellip = Ellipse((x0, y0), 2*a, 2*b, 180/math.pi * th, facecolor='None',
                     edgecolor='grey', linestyle='-', linewidth=2)
    plt.gca().add_patch(ellip)
    plt.plot(x0, y0, color='grey', marker='o', markersize=6)


def loss_rmse(error):
    return torch.sqrt(torch.mean((error)**2))


def error_dist(x0, y0, a, b, th):

    (xc, yc) = (x - x0, y - y0)
    term1 = torch.add( torch.mul(xc, torch.cos(th)), torch.mul(yc, torch.sin(th)) )
    term2 = torch.add( torch.mul(xc, torch.sin(th)), torch.mul(yc, -torch.cos(th)) )

    error_dist = torch.div(torch.square(term1), torch.square(a)) + torch.div(torch.square(term2), torch.square(b)) - 1.0
    
    return error_dist


# read in pts3d data from file (saved using CenterPointPublisher)
srcdir = "{0}/local/resources/digit/cloud-ellipse".format(BASE_PATH)
files = sorted(glob.glob("{0}/*.json".format(srcdir)))
pts3d = np.empty((0, 3), np.float32)
for idx in [0]:
    with open(files[idx]) as f:
        data = json.load(f)
        pts3d_curr = np.asarray(data['pts3d'], dtype=np.float32)  # n x 3
        pts3d = np.vstack((pts3d, pts3d_curr))

npts = pts3d.shape[0]
x = torch.from_numpy(pts3d[:, 0])
y = torch.from_numpy(pts3d[:, 1])

x0 = torch.tensor([0.0])
y0 = torch.tensor([0.0])
a = torch.tensor([0.16/2])
b = torch.tensor([0.24/2])
th = torch.tensor([0.00])

(x0, y0, a, b, th) = Variable(x0, requires_grad=True), Variable(y0, requires_grad=True), Variable(
    a, requires_grad=True), Variable(b, requires_grad=True), Variable(th, requires_grad=True)

params = list([x0, y0, a, b, th])
optimizer = torch.optim.Adam(params, lr=0.002)

fig1 = plt.figure(figsize=(10, 8))
niters = 200
loss = None
for iter in range(niters):
    print("iter: {0}, loss: {1}, (x0, y0, 2a, 2b, th): ({2}, {3}, {4}, {5}, {6})".format(
        iter, loss, x0.item(), y0.item(), 2*a.item(), 2*b.item(), th.item()))

    plt.clf()
    draw_ellip_with_points(x.data.numpy(), y.data.numpy(
    ), x0.data.numpy(), y0.data.numpy(), a.data.numpy(), b.data.numpy(), th.data.numpy())
    plt.pause(1e-3)

    error = error_dist(x0, y0, a, b, th)
    loss = loss_rmse(error)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
