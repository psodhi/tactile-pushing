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

        self.obj_size_x = params['obj_size_x']
        self.obj_size_y = params['obj_size_y']
        self.obj_verts = params['obj_verts']

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

        offset_x = int(0.5 * (self.grid_size_x - self.obj_size_x))
        offset_y = int(0.5 * (self.grid_size_y - self.obj_size_y))

        self.grid_occ[offset_y:(offset_y+self.obj_size_y),
                      offset_x:(offset_x+self.obj_size_x)] = 1

        # self.visualize_grid(self.grid_occ)

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


def set_sdf_params(data):

    params = {}

    params['res'] = 0.001

    # block object (read params from pybullet dataset file)
    params_dataset = data['params']
    params['obj_size_x'] = int(params_dataset['block_size_x'] / params['res'])
    params['obj_size_y'] = int(params_dataset['block_size_y'] / params['res'])
    params['obj_verts'] = np.asarray(data['obj_poly_shape'])  # nverts x 2

    params['grid_size_x'] = 5 * params['obj_size_x']  # cols
    params['grid_size_y'] = 5 * params['obj_size_y']  # rows
    params['origin_x'] = -0.5 * params['grid_size_x'] * params['res']
    params['origin_y'] = -0.5 * params['grid_size_y'] * params['res']

    return params

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def main():

    # read sim pushing dataset file
    dataset_name = "fbDatasetTraj3"
    srcfile = "{0}/local/datasets/{1}.json".format(base_path, dataset_name)
    with open(srcfile) as f:
        data = json.load(f)

    # generate sdf map
    params = set_sdf_params(data)
    sdf = MapSDF(params)

    # save sdf map to file
    dstfile = "{0}/local/datasets/{1}ObjSDF.json".format(base_path, dataset_name)
    sdf.save_map(dstfile)


if __name__ == "__main__":
    main()
