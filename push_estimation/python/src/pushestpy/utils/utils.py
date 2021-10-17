#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import gtsam

import os
import pickle as pkl
from subprocess import call

def vec3_to_pose2(vec):
    return gtsam.Pose2(vec[0], vec[1], vec[2])

def pose2_to_vec3(pose2):
    return [pose2.x(), pose2.y(), pose2.theta()]

def write_video_ffmpeg(imgsrcdir, viddst, framerate=30):

    cmd = "ffmpeg -y -r {0} -pattern_type glob -i '{1}/*.png' {2}.mp4".format(
        framerate, imgsrcdir, viddst)
    call(cmd, shell=True)

def make_dir(dir, clear_flag=False):
    print("Creating directory {0}".format(dir))
    cmd = "mkdir -p {0}".format(dir)
    os.popen(cmd, 'r')

    if clear_flag:
        cmd = "rm -rf {0}/*".format(dir)
        os.popen(cmd, 'r')

def load_pkl_obj(filename):
    with (open(filename, "rb")) as f:
        pkl_obj = pkl.load(f)
    f.close()

    return pkl_obj
