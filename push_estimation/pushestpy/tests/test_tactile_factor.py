#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import unittest
import os

import pushestcpp
import gtsam

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class TestTactileRelativeFactor(unittest.TestCase):

    torch_model_name = '10-25-2020-16-40-01_tf_regr_model_ser_epoch030'
    torch_model_file = f"{BASE_PATH}/local/models/{torch_model_name}.pt"

    theta = np.pi/6
    tf_model_pred =  np.array([0., 0., np.cos(theta), np.sin(theta)]) # tx, ty, cos(\theta), sin(\theta)

    noise_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([1., 1., 1.]))
    factor = pushestcpp.TactileRelativeTfFactor(gtsam.symbol(ord('o'), 0), gtsam.symbol(ord('o'), 1), 
                gtsam.symbol(ord('e'), 0), gtsam.symbol(ord('e'), 1), tf_model_pred, noise_model)

    def test_error_tactile_1(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(0., 0., 0.)
        ee_pose_1 = gtsam.Pose2(0., 0., 0.)
        
        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0., 0., 0.52359878])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_tactile_2(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(1., 1., np.pi/2)
        ee_pose_0 = gtsam.Pose2(0., 0., np.pi/2)
        ee_pose_1 = gtsam.Pose2(1., 1., np.pi)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0., 0., 0.52359878])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_tactile_3(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(1., 1., np.pi/4)
        ee_pose_0 = gtsam.Pose2(1., 1., np.pi/4)
        ee_pose_1 = gtsam.Pose2(2., 2., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0.66650618, 0.86860776, 0.52359878])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_tactile_4(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(1., 1., np.pi/4)
        ee_pose_0 = gtsam.Pose2(2., 2., np.pi/4)
        ee_pose_1 = gtsam.Pose2(3., 3., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([1.33301235, 1.73721552, 0.52359878])
                
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_tactile_5(self):
        obj_pose_0 = gtsam.Pose2(1., 1., np.pi/2)
        obj_pose_1 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(2., 2., np.pi/4)
        ee_pose_1 = gtsam.Pose2(3., 3., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([-4.64499601, -2.25899128, -1.83259571])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

if __name__ == '__main__':
    unittest.main()