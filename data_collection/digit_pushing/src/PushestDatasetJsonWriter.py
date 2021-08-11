#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import rospy
import rospkg

import cv2
import tf
import io
import os

import numpy as np
import json

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class PushestDatasetJsonWriter:
    def __init__(self):

        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/digit/digit_alpha/image_raw/", Image, self.callback_digit_image)

        # contact episodes related vars
        self.contact_episode_idx = 0
        self.counter = 0
        self.num_incontact = 0
        self.min_num_incontact = 5

        # to be logged data
        self.ee_pose2d_list = []
        self.obj_pose2d_list = []
        self.contact_episode_list = []
        self.contact_flag_list = []

        self.init_dataset_params()

        self.dstdir = rospy.get_param("dstdir_dataset")
        self.bag_name = rospy.get_param("bag_name")
        rospy.loginfo(
            "[PushestDatasetJsonWriter] Using bag {0}.bag".format(self.bag_name))

        rospack = rospkg.RosPack()
        self.path_pkg = rospack.get_path("digit_pushing")
        self.mean_img = cv2.imread(
            "{0}/local/resources/digit/{1}/mean_img.png".format(self.path_pkg, self.bag_name)).astype(np.float32)
        self.std_img = cv2.imread(
            "{0}/local/resources/digit/{1}/std_img.png".format(self.path_pkg, self.bag_name)).astype(np.float32)

    def make_dir(self, dir):
        cmd = "mkdir -p {0}".format(dir)
        os.popen(cmd, 'r')

    def in_contact(self, img):

        # Compute per-image sum of stddev squared
        diff = np.linalg.norm((img - self.mean_img)/self.std_img)**2
        diff = diff / self.mean_img.size

        # Count the percent of pixels that are significantly different from their mean values
        diff_cnt = np.sum(((img - self.mean_img)/self.std_img)**2 > 4**2)
        diff_cnt = float(diff_cnt) / float(self.mean_img.size)

        # contact_flag = diff_cnt > 0.05
        contact_flag = diff_cnt > 0.01

        return contact_flag

    def rosimg_to_numpy(self, imgmsg):
        if hasattr(imgmsg, 'format') and 'compressed' in imgmsg.format:
            return np.asarray(Image.open(io.BytesIO(imgmsg.data)))

        return np.frombuffer(imgmsg.data, dtype=np.uint8).reshape(imgmsg.height, imgmsg.width, 3)[:, :, ::-1]

    def remove_contact_episode(self, episode_idx):
        indices = [idx for idx, elem in enumerate(
            self.contact_episode_list) if elem == 4]
        for i in sorted(indices, reverse=True):
            del self.contact_episode_list[i]
            del self.obj_pose2d_list[i]
            del self.ee_pose2d_list[i]

    def init_dataset_params(self):
        self.params = {}
        self.params['obj_radius'] = 0.088

    def save_data2d_json(self):

        data = {'params': self.params,
                'ee_poses_2d': self.ee_pose2d_list,
                'obj_poses_2d': self.obj_pose2d_list,
                'contact_flag': self.contact_flag_list,
                'contact_episode': self.contact_episode_list}

        dstfile = "{0}/{1}_{2}.json".format(self.dstdir,
                                            self.bag_name, self.contact_episode_idx)
        with open(dstfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        rospy.loginfo("Wrote json dataset for episodes 0 to {0} at:\n {1} ".format(
            self.contact_episode_idx, dstfile))

    def callback_digit_image(self, msg):

        try:
            # img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = self.rosimg_to_numpy(msg)
        except CvBridgeError as e:
            rospy.logwarn(
                "[PushestDatasetJsonWriter::callback_digit_image] {0}".format(e))
            return

        try:
            # looks up arg2 frame transform in arg1 frame
            (trans_obj, rot_obj) = self.tf_listener.lookupTransform(
                "world", "/object/center/", rospy.Time(0))
            (trans_ee, rot_ee) = self.tf_listener.lookupTransform(
                "world", "/digit/center/", rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(
                "[PushestDatasetJsonWriter::callback_digit_image] TF lookup failed")
            return

        if (self.in_contact(img)):

            rot_obj_euler = tf.transformations.euler_from_quaternion(rot_obj)
            obj_pose2d = [trans_obj[0], trans_obj[1],
                          rot_obj_euler[2]]  # (x, y, yaw)

            rot_ee_euler = tf.transformations.euler_from_quaternion(rot_ee)
            ee_pose2d = [trans_ee[0], trans_ee[1],
                         rot_ee_euler[2]]  # (x, y, yaw)

            # add to data list being logged
            self.obj_pose2d_list.append(obj_pose2d)
            self.ee_pose2d_list.append(ee_pose2d)
            self.contact_flag_list.append([1])
            self.contact_episode_list.append([self.contact_episode_idx])

            self.num_incontact = self.num_incontact + 1

        else:
            self.counter = self.counter + 1

        # start new contact episode
        if ((self.counter > 10) & (self.num_incontact > 1)):

            if (self.num_incontact > self.min_num_incontact):
                self.save_data2d_json()
                self.contact_episode_idx = self.contact_episode_idx + 1
            else:
                self.remove_contact_episode(self.contact_episode_idx)

            self.counter = 0
            self.num_incontact = 0


def main():
    img_tf_writer = PushestDatasetJsonWriter()
    rospy.init_node('pushest_dataset_json_writer', anonymous=True)
    rospy.loginfo("Initialized pushest_dataset_json_writer node.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
