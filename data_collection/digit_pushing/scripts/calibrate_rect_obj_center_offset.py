#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import json
import glob
import os

plt.rcParams.update({'font.size': 18})
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def draw_rect_with_points(x, y, x0, y0, w, h):

    plt.ion()
    plt.scatter(x, y, color="red", label="rect-pts")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')

    plt.show()

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    rect = Rectangle((x0-0.5*w, y0-0.5*h), w, h, facecolor='None',
                     edgecolor='grey', linestyle='-', linewidth=2)
    plt.gca().add_patch(rect)
    plt.plot(x0, y0, color='grey', marker='o', markersize=6)


def loss_rmse(error):
    return torch.sqrt(torch.mean((error)**2))


def error_dist(x0, y0, w, h):
    d1 = torch.square(x - x0 - w/2)[None, :]
    d2 = torch.square(x - x0 + w/2)[None, :]
    d3 = torch.square(y - y0 - h/2)[None, :]
    d4 = torch.square(y - y0 + h/2)[None, :]

    dist = torch.cat((d1, d2, d3, d4), 0)
    error_dist, min_idxs = torch.min(dist, 0)

    return error_dist


# read in pts3d data from file (saved using CenterPointPublisher)
srcdir = "{0}/local/resources/digit/cloud-rect".format(BASE_PATH)
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

x0 = torch.tensor(np.array([0.0]), requires_grad=True)
y0 = torch.tensor(np.array([0.0]), requires_grad=True)
w = torch.tensor(np.array([0.24]), requires_grad=True)
h = torch.tensor(np.array([0.16]), requires_grad=True)

(x0, y0, w, h) = Variable(x0, requires_grad=True), Variable(
    y0, requires_grad=True), Variable(w, requires_grad=True), Variable(h, requires_grad=True)

params = list([x0, y0, w, h])
optimizer = torch.optim.Adam(params, lr=0.002)

fig1 = plt.figure(figsize=(10, 8))
niters = 200
loss = None
for iter in range(niters):
    print("iter: {0}, loss: {1}, (x0, y0, w, h): ({2}, {3}, {4}, {5})".format(
        iter, loss, x0, y0, w, h))

    plt.clf()
    draw_rect_with_points(x.data.numpy(), y.data.numpy(
    ), x0.data.numpy(), y0.data.numpy(), w.data.numpy(), h.data.numpy())
    plt.pause(1e-3)

    error = error_dist(x0, y0, w, h)
    loss = loss_rmse(error)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
