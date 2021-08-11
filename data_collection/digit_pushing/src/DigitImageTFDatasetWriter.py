#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import rospy
import rospkg

import tf
import io
import os
import imageio

import numpy as np
import json

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DigitImageTFDatasetWriter:
    def __init__(self):

        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/digit/digit_alpha/image_raw/", Image, self.callback_digit_image)

        self.contact_episode_idx = 0
        self.counter = 0
        self.img_idx = 0
        self.min_num_img = 3

        self.ee_poses2d__obj_list = []
        self.obj_poses2d__world_list = []
        self.ee_poses2d__world_list = []

        self.dstdir = rospy.get_param("dstdir_dataset")
        self.make_dir(
            "{0}/episode_{1:04d}/".format(self.dstdir, self.contact_episode_idx))

        self.bag_name = rospy.get_param("bag_name")
        rospy.loginfo(
            "[DigitImageTFDatasetWriter::init] Using bag {0}.bag".format(self.bag_name))

        # image contact detector threshold
        self.obj_shape = rospy.get_param("obj_shape")
        if (self.obj_shape == "disc"):
            self.contact_thresh = 0.01
        elif (self.obj_shape == "rectangle"):
            self.contact_thresh = 0.1
        elif (self.obj_shape == "ellipse"):
            self.contact_thresh = 0.1
        else:
            rospy.logwarn("contact_thresh not defined for object shape {0}. Using default value.".format(self.obj_shape))

        # read mean, std images from file for the particular bag dataset
        rospack = rospkg.RosPack()
        self.path_pkg = rospack.get_path('digit_pushing')
        filename = "{0}/local/resources/digit/{1}/mean_std_img.json".format(self.path_pkg, self.bag_name)
        with open(filename) as f:
            data = json.load(f)
            self.mean_img = np.asarray(data['mean_img'], dtype=np.float32)
            self.std_img = np.asarray(data['std_img'], dtype=np.float32) 

    def make_dir(self, dir):
        cmd = "mkdir -p {0}".format(dir)
        os.popen(cmd, 'r')
    
    def remove_dir(self, dir):
        cmd = "rm -r {0}".format(dir)
        os.popen(cmd, 'r')

    def save_poses2d_json(self):
        data = {'ee_poses2d__obj': self.ee_poses2d__obj_list,
                'obj_poses2d__world': self.obj_poses2d__world_list,
                'ee_poses2d__world': self.ee_poses2d__world_list}

        outfile_poses = "{0}/episode_{1:04d}/poses2d_episode_{2:04d}.json".format(
            self.dstdir, self.contact_episode_idx, self.contact_episode_idx)
        with open(outfile_poses, 'w') as outfile_tf:
            json.dump(data, outfile_tf, indent=4)

        rospy.loginfo("Wrote images, poses for contact episode {0} to: \n {1}/episode_{2:04d}".format(
            self.contact_episode_idx, self.dstdir, self.contact_episode_idx))

    def remove_contact_episode(self, episode_idx):
        self.remove_dir(
        "{0}/episode_{1:04d}/".format(self.dstdir, self.contact_episode_idx))

    def in_contact(self, img):

        # Compute per-image sum of stddev squared
        diff = np.linalg.norm((img - self.mean_img)/self.std_img)**2
        diff = diff / self.mean_img.size

        # Count the percent of pixels that are significantly different from their mean values
        diff_cnt = np.sum(((img - self.mean_img)/self.std_img)**2 > 4**2)
        diff_cnt = float(diff_cnt) / float(self.mean_img.size)

        contact_flag = diff_cnt > self.contact_thresh

        return contact_flag

    def rosimg_to_numpy(self, imgmsg):
        if hasattr(imgmsg, 'format') and 'compressed' in imgmsg.format:
            return np.asarray(Image.open(io.BytesIO(imgmsg.data)))

        return np.frombuffer(imgmsg.data, dtype=np.uint8).reshape(imgmsg.height, imgmsg.width, 3)[:, :, ::-1]

    def callback_digit_image(self, msg):

        try:
            # img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = self.rosimg_to_numpy(msg)
        except CvBridgeError as e:
            rospy.logwarn(
                "[DigitImageTFDatasetWriter::callback_digit_image] {0}".format(e))
            return

        try:
            # looks up arg2 frame transform in arg1 frame
            (trans_ee__obj, rot_ee__obj) = self.tf_listener.lookupTransform(
                "/object/center/", "/digit/center/", rospy.Time(0))
            (trans_obj__world, rot_obj__world) = self.tf_listener.lookupTransform(
                "world", "/object/center/", rospy.Time(0))
            (trans_ee__world, rot_ee__world) = self.tf_listener.lookupTransform(
                "world", "/digit/center/", rospy.Time(0))
                        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(
                "[DigitImageTFDatasetWriter::callback_digit_image] TF lookup failed")
            return

        if (self.in_contact(img)):

            # save image to file
            outfile_img = "{0}/episode_{1:04d}/{2:04d}.png".format(
                self.dstdir, self.contact_episode_idx, self.img_idx)

            # cv2.imshow("img", img)
            # cv2.waitKey(5)
            # plt.imshow(img.astype(np.uint8))
            # plt.pause(1e-3)

            imageio.imwrite(outfile_img, img)

            # append pose to list (saved to file at end of episode)

            # a. digit pose in object frame: [x, y, yaw]
            rot_ee__obj_euler = tf.transformations.euler_from_quaternion(rot_ee__obj)
            ee_pose2d__obj = [trans_ee__obj[0], trans_ee__obj[1], rot_ee__obj_euler[2]]

            # b. obj pose in world frame: [x, y, yaw]
            rot_obj__world_euler = tf.transformations.euler_from_quaternion(rot_obj__world)
            obj_pose2d__world = [trans_obj__world[0], trans_obj__world[1], rot_obj__world_euler[2]]

            # c. digit pose in world frame: [x, y, yaw]
            rot_ee__world_euler = tf.transformations.euler_from_quaternion(rot_ee__world)
            ee_pose2d__world = [trans_ee__world[0], trans_ee__world[1], rot_ee__world_euler[2]]

            self.ee_poses2d__obj_list.append(ee_pose2d__obj)
            self.obj_poses2d__world_list.append(obj_pose2d__world)
            self.ee_poses2d__world_list.append(ee_pose2d__world)

            self.img_idx = self.img_idx + 1

        else:
            self.counter = self.counter + 1

        # end of contact episode
        if ((self.counter > 10) & (self.img_idx > 1)):

            if (self.img_idx > self.min_num_img):
                self.save_poses2d_json()
                self.contact_episode_idx = self.contact_episode_idx + 1
                self.make_dir(
                    "{0}/episode_{1:04d}/".format(self.dstdir, self.contact_episode_idx))
            # else:
                # self.remove_contact_episode(self.contact_episode_idx)

            # reset vars
            self.counter = 0
            self.img_idx = 0

            self.ee_poses2d__obj_list = []
            self.obj_poses2d__world_list = []
            self.ee_poses2d__world_list = []

def main():
    rospy.init_node('digit_img_tf_dataset_writer', anonymous=True)
    rospy.loginfo("Initialized digit_img_tf_dataset_writer node.")

    img_tf_writer = DigitImageTFDatasetWriter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
