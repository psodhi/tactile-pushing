#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import json

import scipy.ndimage.morphology as scimorph
import os

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

        self.obj_radius = params['obj_radius']
        self.obj_size_x = params['obj_size_x']
        self.obj_size_y = params['obj_size_y']

        # occupancy, distance field grids
        self.grid_occ = np.zeros((self.grid_size_y, self.grid_size_x))
        self.grid_sdf = np.zeros((self.grid_size_y, self.grid_size_x))

        self.compute_grid_sdf()

    def visualize_grid(self, grid):

        cmap = colors.Colormap('Sequential')
        img = plt.imshow(grid, interpolation='nearest')
        plt.colorbar(img, cmap=cmap)

        plt.show()

    def compute_grid_occ(self):
        
        # import pdb; pdb.set_trace()

        offset_x = int(0.5 * (self.grid_size_x - self.obj_size_x))
        offset_y = int(0.5 * (self.grid_size_y - self.obj_size_y))

        grid_center_x = int(0.5*self.grid_size_x)
        grid_center_y = int(0.5*self.grid_size_y)
        radius = self.obj_radius
        for grid_y in range(offset_y, offset_y+self.obj_size_y): # row-wise traversal
            dist_y = self.res * np.absolute(grid_y - grid_center_y)
            theta = np.arccos(dist_y/radius)

            grid_x_min = grid_center_x - int(radius * np.sin(theta) / self.res)
            grid_x_max = grid_center_x + int(radius * np.sin(theta) / self.res)

            self.grid_occ[grid_y, grid_x_min:grid_x_max] = 1

        self.visualize_grid(self.grid_occ)

    def compute_grid_sdf(self):

        self.compute_grid_occ()

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


def set_sdf_params():

    params = {}

    params['res'] = 0.001
    params['obj_radius'] = 0.088
    params['obj_size_x'] = int(2 * params['obj_radius'] / params['res'])
    params['obj_size_y'] = int(2 * params['obj_radius'] / params['res'])

    params['grid_size_x'] = 8 * params['obj_size_x']  # cols
    params['grid_size_y'] = 8 * params['obj_size_y']  # rows
    params['origin_x'] = -0.5 * params['grid_size_x'] * params['res']
    params['origin_y'] = -0.5 * params['grid_size_y'] * params['res']

    return params

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

def main():

    # read real pushing dataset file
    # dataset_name = "fbDatasetTraj3"
    # srcfile = "{0}/local/data/{1}.json".format(base_path, dataset_name)
    # with open(srcfile) as f:
        # data = json.load(f)

    shape_name = "disc"

    # generate sdf map
    params = set_sdf_params()
    sdf = MapSDF(params)

    # save sdf map to file
    dstfile = "{0}/local/shapes/{1}.json".format(BASE_PATH, shape_name)
    sdf.save_map(dstfile)


if __name__ == "__main__":
    main()
