#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import json

import scipy.ndimage.morphology as scimorph
import os
import cv2

import matplotlib.pyplot as plt
from matplotlib import colors

class MapSDF():
    def __init__(self, params):

        # params
        self.res = params['res']

        self.grid_size_x = params['grid_size_x']
        self.grid_size_y = params['grid_size_y']
        self.origin_x = params['origin_x']
        self.origin_y = params['origin_y']

        self.obj_size_x = params['obj_size_x']
        self.obj_size_y = params['obj_size_y']

        # occupancy, distance field grids
        self.img_occ = params['img_occ'] 
        self.grid_occ = np.zeros((self.grid_size_y, self.grid_size_x))
        self.grid_sdf = np.zeros((self.grid_size_y, self.grid_size_x))

        self.compute_grid_sdf()

    def visualize_grid(self, grid):

        cmap = colors.Colormap('Sequential')
        img = plt.imshow(-grid, interpolation='nearest', cmap='YlOrRd')
        plt.colorbar(img, cmap=cmap)

        plt.show()

    def compute_grid_occ_from_img(self):
        self.grid_occ = cv2.resize(self.img_occ , (self.obj_size_x, self.obj_size_y))

        # occupied cells -> 1, free cells -> 0
        self.grid_occ = np.where(self.grid_occ < 255, 1, self.grid_occ)
        self.grid_occ = np.where(self.grid_occ == 255, 0, self.grid_occ)

        offset_x = int(0.5 * (self.grid_size_x - self.obj_size_x))
        offset_y = int(0.5 * (self.grid_size_y - self.obj_size_y))
        self.grid_occ = cv2.copyMakeBorder(self.grid_occ, offset_y, offset_y, offset_x, offset_x, 
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.visualize_grid(self.grid_occ)

    def compute_grid_sdf(self):

        self.compute_grid_occ_from_img()

        map_dist = scimorph.distance_transform_edt(self.grid_occ)
        inv_map_dist = scimorph.distance_transform_edt(1-self.grid_occ)

        self.grid_sdf = inv_map_dist - map_dist
        self.grid_sdf = self.grid_sdf * self.res

        self.visualize_grid(self.grid_sdf)

    def save_map(self, filename):

        data = {'grid_res': self.res,
                'grid_size_x': self.grid_size_x,
                'grid_size_y': self.grid_size_y,
                'grid_origin_x': self.origin_x,
                'grid_origin_y': self.origin_y,
                'grid_data': self.grid_sdf.tolist()}

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        print("Finished writing object sdf to {0}".format(filename))

def visualize_grid(grid):

    cmap = colors.Colormap('Sequential')
    img = plt.imshow(grid, interpolation='nearest')
    plt.colorbar(img, cmap=cmap)

    plt.show()

def set_sdf_params(shape_name):

    params = {}

    params['res'] = 0.001

    if (shape_name == "rect"):
        params['obj_size_x'] = int(0.2363 / params['res'])
        params['obj_size_y'] = int(0.1579 / params['res'])
    elif (shape_name == "ellip"):
        params['obj_size_x'] = int(0.1638 / params['res'])
        params['obj_size_y'] = int(0.2428 / params['res'])

    params['grid_size_x'] = 5 * params['obj_size_x']  # cols
    params['grid_size_y'] = 5 * params['obj_size_y']  # rows
    params['origin_x'] = -0.5 * params['grid_size_x'] * params['res']
    params['origin_y'] = -0.5 * params['grid_size_y'] * params['res']

    imgfile_occ = "{0}/local/shapes/{1}_occ_img.png".format(BASE_PATH, shape_name)
    params['img_occ'] = cv2.imread(imgfile_occ, cv2.IMREAD_GRAYSCALE)

    return params

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

def main():

    # shape_name = "rounded_rect"
    shape_name = "ellip"

    # generate sdf map
    params = set_sdf_params(shape_name)
    sdf = MapSDF(params)

    # save sdf map to file
    dstfile = "{0}/local/shapes/{1}.json".format(BASE_PATH, shape_name)
    sdf.save_map(dstfile)


if __name__ == "__main__":
    main()
