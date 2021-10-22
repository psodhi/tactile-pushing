#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import unittest

import gtsam

class TestPose2(unittest.TestCase):

    def print(self, expected, actual):
        print(f"Expected: {expected}, Actual: {actual}")

    def test_error_pose2_1(self):

        pose_1 = gtsam.Pose2(1., 2., np.pi/2.0)
        pose_2 = gtsam.Pose2(1.015, 2.01, np.pi/2.0+0.18)

        actual = pose_1.localCoordinates(pose_2)

        # expected = np.array([0.01, -0.015, 0.018])
        expected = np.array([0.00862299, -0.0150896, 0.018]) # SLOW_BUT_CORRECT_EXPMAP

        self.print(expected, actual)
        self.assertEqual(np.allclose(actual, expected, atol=1e-5), True)

if __name__ == '__main__':
    unittest.main()