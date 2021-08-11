#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

from subprocess import call

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
import json
import glob
import os

plt.rcParams.update({'font.size': 18})
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def RMSELoss(error):
    return torch.sqrt(torch.mean((error)**2))


def write_video(imgsrcdir, viddst):

    framerate = 15
    cmd = "ffmpeg -y -r {0} -pattern_type glob -i '{1}/*.png' {2}.mp4".format(
        framerate, imgsrcdir, viddst)
    call(cmd, shell=True)

    framerate = 30
    cmd = "convert -quality 0.10 -layers Optimize -delay {0} -loop 0 {1}/*.png {2}.gif".format(
        framerate, imgsrcdir, viddst)
    call(cmd, shell=True)


def draw_circle_with_points(x, y, x0, y0, r):

    plt.ion()
    plt.scatter(x, y, color="red", label="disk-pts")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')

    plt.show()

    plt.xlim([-0.15, 0.15])
    plt.ylim([-0.15, 0.15])

    circ = Circle((x0, y0), r, facecolor='None',
                  edgecolor='grey', linestyle='-', linewidth=2)
    plt.gca().add_patch(circ)


# read in pts3d data from file (saved using CenterPointPublisher)
srcdir = "{0}/local/resources/digit/cloud-disc".format(BASE_PATH)
files = sorted(glob.glob("{0}/*.json".format(srcdir)))
pts3d = np.empty((0, 3), np.float32)
for idx in [0, 2]:
    with open(files[idx]) as f:
        data = json.load(f)
        pts3d_curr = np.asarray(data['pts3d'], dtype=np.float32)  # n x 3
        pts3d = np.vstack((pts3d, pts3d_curr))

npts = pts3d.shape[0]
x = torch.from_numpy(pts3d[:, 0])
y = torch.from_numpy(pts3d[:, 1])

# view data
fig1 = plt.figure(figsize=(10, 8))

x0 = torch.zeros((1), requires_grad=True)
y0 = torch.zeros((1), requires_grad=True)
r = torch.tensor(np.array([0.0762]), requires_grad=True)

(x0, y0, r) = Variable(x0, requires_grad=True), Variable(
    y0, requires_grad=True), Variable(r, requires_grad=True)

# test estimated params on a different dataset
# draw_circle_with_points(x.data.numpy(), y.data.numpy(), 0.0103, 0.0102, 0.0910)

save_vid = False


niters = 50
params = list([x0, y0, r])
optimizer = torch.optim.Adam(params, lr=0.002)

for iter in range(niters):
    error = torch.square(x-x0) + torch.square(y-y0) - torch.square(r)
    loss = RMSELoss(error)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("iter: {0}, loss: {1}, (x0, y0, r): ({2}, {3}, {4})".format(
        iter, loss, x0, y0, r))

    plt.clf()
    draw_circle_with_points(x.data.numpy(), y.data.numpy(
    ), x0.data.numpy(), y0.data.numpy(), r.data.numpy())
    plt.pause(1e-3)

    if save_vid:
        dstdir = "{0}/local/visualizations/cloud_disc_regression".format(
            BASE_PATH)
        figfile = "{0}/{1:06d}.png".format(dstdir, iter)
        plt.savefig(figfile)

if save_vid:
    imgsrcdir = "{0}/local/visualizations/cloud_disc_regression".format(
        BASE_PATH)
    vidname = "disc_regression"
    viddst = "{0}/local/visualizations/{1}".format(BASE_PATH, vidname)
    write_video(imgsrcdir, viddst)
