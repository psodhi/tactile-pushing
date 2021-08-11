#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import rospy
import rospkg
import tf
import io

import numpy as np
import json

from sensor_msgs.msg import PointCloud
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point32
from jsk_rviz_plugins.msg import OverlayText

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CenterPointPublisher:
    def __init__(self):
        # point cloud publisher
        self.cloud_pub = rospy.Publisher(
            "/digit/center/cloud", PointCloud, queue_size=1)

        # image subscriber, tf listener
        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/digit/digit_alpha/image_raw/", Image, self.callback_digit_image)
        
        # contact flag publisher
        self.contact_flag_pub = rospy.Publisher("/digit/contact/flag", OverlayText, queue_size=1)

        # image contact detector threshold
        self.obj_shape = rospy.get_param("obj_shape")
        self.contact_thresh = 0.05
        if (self.obj_shape == "disc"):
            self.contact_thresh = 0.01
        elif (self.obj_shape == "rectangle"):
            self.contact_thresh = 0.1
        elif (self.obj_shape == "ellipse"):
            self.contact_thresh = 0.1
        else:
            rospy.logwarn("contact_thresh not defined for object shape {0}. Using default value.".format(self.obj_shape))

        # accumulate points over timesteps (in contact)
        self.cloud_msg = PointCloud()
        self.cloud_z_offset = 0.02

        self.bag_name = rospy.get_param("bag_name")
        rospy.loginfo("[CenterPointsPublisher] Using bag {0}.bag".format(self.bag_name))

        self.save_json = False
        self.pts3d_json = []

        rospack = rospkg.RosPack()
        self.path_pkg = rospack.get_path('digit_pushing')
        filename = "{0}/local/resources/digit/{1}/mean_std_img.json".format(self.path_pkg, self.bag_name)
        with open(filename) as f:
            data = json.load(f)
            self.mean_img = np.asarray(data['mean_img'], dtype=np.float32)
            self.std_img = np.asarray(data['std_img'], dtype=np.float32) 
                
    def in_contact(self, img):

        # Compute per-image sum of stddev squared
        diff = np.linalg.norm((img - self.mean_img)/self.std_img)**2
        diff = diff / self.mean_img.size

        # Count the percent of pixels that are significantly different from their mean values
        diff_cnt = np.sum(((img - self.mean_img)/self.std_img)**2 > 4**2)
        diff_cnt = float(diff_cnt) / float(self.mean_img.size)

        contact_flag = diff_cnt > self.contact_thresh

        # rospy.loginfo("diff_cnt: {0}, contact_flag: {1}\n".format(diff_cnt, contact_flag))
        contact_flag_msg = OverlayText()
        contact_flag_msg.text = "diff_cnt: {0:03f}\n contact_flag: {1}".format(diff_cnt, contact_flag)
        self.contact_flag_pub.publish(contact_flag_msg)

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
                "[CenterPointsPublisher::callback_digit_image] {0}".format(e))
            return

        try:
            # looks up child frame transform in parent frame
            # parent_frame = "world"
            parent_frame = "/object/center/"
            child_frame = "/digit/center/"
            (trans, rot) = self.tf_listener.lookupTransform(
                parent_frame, child_frame, rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(
                "[CenterPointsPublisher::callback_digit_image] TF lookup failed")
            return


        if (self.in_contact(img)):
            header = msg.header
            header.frame_id = parent_frame

            self.cloud_msg.header = header
            self.cloud_msg.points.append(Point32(trans[0], trans[1], trans[2]+self.cloud_z_offset))
            self.cloud_pub.publish(self.cloud_msg)

            if (self.save_json):
                self.pts3d_json.append([trans[0], trans[1], trans[2]])

                dstdir = "{0}/local/resources/digit/cloud-{1}/".format(
                    self.path_pkg, self.obj_shape)
                # cmd = "mkdir -p {0}".format(dstdir)

                outfile = "{0}/{1}.json".format(dstdir, self.bag_name)
                data = {'pts3d': self.pts3d_json}
                with open(outfile, 'w') as outfile:
                    json.dump(data, outfile, indent=4)


def main():
    rospy.init_node("center_point_publisher")
    rospy.loginfo("Initialized center_points_publisher node.")

    center_pts_pub = CenterPointPublisher()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':

    main()
