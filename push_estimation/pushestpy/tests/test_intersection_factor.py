#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import unittest

import os
import json

import pushestcpp
import gtsam

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def load_object_sdf(sdf_file):
    with open(sdf_file) as f:
        sdf_data = json.load(f)

    cell_size = sdf_data['grid_res']
    sdf_cols = sdf_data['grid_size_x']
    sdf_rows = sdf_data['grid_size_y']
    sdf_data_vec = sdf_data['grid_data']
    sdf_origin_x = sdf_data['grid_origin_x']
    sdf_origin_y = sdf_data['grid_origin_y']

    origin = gtsam.Point2(sdf_origin_x, sdf_origin_y)

    sdf_data_mat = np.zeros((sdf_rows, sdf_cols))
    for i in range(sdf_data_mat.shape[0]):
        for j in range(sdf_data_mat.shape[1]):
            sdf_data_mat[i, j] = sdf_data_vec[i][j]

    object_sdf = pushestcpp.PlanarSDF(origin, cell_size, sdf_data_mat)

    return object_sdf

class TestIntersectionFactor(unittest.TestCase):

    ee_radius = 0.1
    noise_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([1., 1., 1.]))

    sdf_file = f'{BASE_PATH}/local/sdfs/rect.json'
    object_sdf = load_object_sdf(sdf_file)
    factor = pushestcpp.IntersectionPlanarSDFObjEEFactor(
                gtsam.symbol(ord('o'), 0), gtsam.symbol(ord('e'), 0), object_sdf, ee_radius, noise_model)
    
    def print(self, expected, actual):
        print(f"Expected: {expected}, Actual: {actual}")

    def test_error_intersection_1(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(0., 0., 0.)

        actual = self.factor.evaluateError(obj_pose_0, ee_pose_0)
        expected = 0.1285000
 
        self.print(expected, actual)
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_intersection_2(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(0., 0., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, ee_pose_0)
        expected = 0.1285000
 
        self.print(expected, actual)
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_intersection_3(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(0.1, 0.2, 0.)

        actual = self.factor.evaluateError(obj_pose_0, ee_pose_0)
        expected = 0.0025903
 
        self.print(expected, actual)
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_intersection_4(self):
        obj_pose_0 = gtsam.Pose2(0.2, 0.1, 0.)
        ee_pose_0 = gtsam.Pose2(0.1, 0.2, np.pi/4)

        actual = self.factor.evaluateError(obj_pose_0, ee_pose_0)
        expected = 0.0300140
 
        self.print(expected, actual)
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_intersection_5(self):
        obj_pose_0 = gtsam.Pose2(0.5, 0.3, 0.)
        ee_pose_0 = gtsam.Pose2(0.2, 0.6, 0.)

        actual = self.factor.evaluateError(obj_pose_0, ee_pose_0)
        expected = 0.2444916
 
        self.print(expected, actual)
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)
        
if __name__ == '__main__':
    unittest.main()