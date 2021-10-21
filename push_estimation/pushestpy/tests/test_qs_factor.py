#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import unittest

import pushestcpp
import gtsam

class TestQuasiStaticFactor(unittest.TestCase):

    # c_sq is c**2, c = max_torque / max_force is a hyperparameter dependent on object
    c_sq = 1.
    noise_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([1., 1., 1.]))
    factor = pushestcpp.QSVelPushMotionRealObjEEFactor(
        gtsam.symbol(ord('o'), 0), gtsam.symbol(ord('o'), 1),
        gtsam.symbol(ord('e'), 0), gtsam.symbol(ord('e'), 1),
        c_sq, noise_model)

    def test_error_qs_1(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(0., 0., 0.)
        ee_pose_1 = gtsam.Pose2(0., 0., 0.)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0., 0., 0.])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_qs_2(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(1., 1., np.pi/2)
        ee_pose_0 = gtsam.Pose2(0., 0., np.pi/2)
        ee_pose_1 = gtsam.Pose2(1., 1., np.pi)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0., 0., -1.57079633])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_qs_3(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(1., 1., np.pi/4)
        ee_pose_0 = gtsam.Pose2(0., 0., np.pi/4)
        ee_pose_1 = gtsam.Pose2(1., 1., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0., 0., -0.78539816])

        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_qs_4(self):
        obj_pose_0 = gtsam.Pose2(0., 0., 0.)
        obj_pose_1 = gtsam.Pose2(1., 1., np.pi/4)
        ee_pose_0 = gtsam.Pose2(2., 2., np.pi/4)
        ee_pose_1 = gtsam.Pose2(3., 3., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([0., 2.22144147, -0.785398163])
        
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

    def test_error_qs_5(self):
        obj_pose_0 = gtsam.Pose2(1., 1., np.pi/2)
        obj_pose_1 = gtsam.Pose2(0., 0., 0.)
        ee_pose_0 = gtsam.Pose2(2., 2., np.pi/4)
        ee_pose_1 = gtsam.Pose2(3., 3., np.pi/2)

        actual = self.factor.evaluateError(obj_pose_0, obj_pose_1, ee_pose_0, ee_pose_1)
        expected = np.array([2.71238898, -6.71238898, 1.57079633])
    
        self.assertEqual(np.allclose(actual, expected, atol=1e-6), True)

if __name__ == '__main__':
    unittest.main()