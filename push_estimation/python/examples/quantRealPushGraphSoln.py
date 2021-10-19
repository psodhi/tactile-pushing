#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys

import hydra
import pickle as pkl
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pushestpy.eval.quant_plotter as qpt


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(
    BASE_PATH, "python/config/eval_real_push_graph.yaml")

@hydra.main(config_path=config_path, strict=False)
def main(cfg):

    if cfg.options.random_seed is not None:
        np.random.seed(cfg.options.random_seed)

    cfg.BASE_PATH = BASE_PATH
    qpt.eval_traj_error(cfg)
    # qpt.runtime_plot(cfg)

if __name__ == "__main__":
    main()
