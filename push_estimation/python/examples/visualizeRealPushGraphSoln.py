#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import hydra
import pickle as pkl
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pushestpy.eval.qual_visualizer as qviz


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(
    BASE_PATH, "python/config/eval_real_push_graph.yaml")


@hydra.main(config_path=config_path, strict=False)
def main(cfg):
    cfg.BASE_PATH = BASE_PATH
    qviz.draw_poses_from_file(cfg)


if __name__ == "__main__":
    main()
