#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import rospy
import tf

from geometry_msgs.msg import PoseStamped

class TFBroadcaster:

    def __init__(self):

        self.obj_shape = rospy.get_param("obj_shape")

        if (self.obj_shape == "disc"):
            self.obj_topic = "Block"
        elif (self.obj_shape == "rectangle"):
            self.obj_topic = "Rectangle"
        elif (self.obj_shape == "ellipse"):
            self.obj_topic = "Ellipse"
        else:
            self.obj_topic = self.obj_shape

        # broadcast tf frames for object, digit mocap pose messages
        self.digit_sub = rospy.Subscriber(
            "/vrpn_client_node/Digit/pose", PoseStamped, self.handler_digit_tf)
        self.object_sub = rospy.Subscriber(
            "/vrpn_client_node/{0}/pose".format(self.obj_topic), PoseStamped, self.handler_object_tf)

        # publish new messages for digit and object centers
        self.digit_center_pub = rospy.Publisher(
            "/vrpn_client_node/Digit/center/pose", PoseStamped, queue_size=1)
        self.digit_pose_sub = rospy.Subscriber(
            "/vrpn_client_node/Digit/pose", PoseStamped, self.callback_digit_pose)
        self.object_center_pub = rospy.Publisher(
            "/vrpn_client_node/{0}/center/pose".format(self.obj_topic), PoseStamped, queue_size=1)
        self.object_pose_sub = rospy.Subscriber(
            "/vrpn_client_node/{0}/pose".format(self.obj_topic), PoseStamped, self.callback_object_pose)

        # broadcast tf frames for the new digit, object center messages
        self.digit_center_sub = rospy.Subscriber(
            "/vrpn_client_node/Digit/center/pose", PoseStamped, self.handler_digit_center_tf)
        self.object_center_sub = rospy.Subscriber(
            "/vrpn_client_node/{0}/center/pose".format(self.obj_topic), PoseStamped, self.handler_object_center_tf)

    def handler_digit_tf(self, msg):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w),
                         msg.header.stamp, "/digit/", msg.header.frame_id)

    def handler_object_tf(self, msg):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w),
                         msg.header.stamp, "/object/", msg.header.frame_id)

    def callback_digit_pose(self, msg):
        digit_center_pose = PoseStamped()

        digit_center_pose.header = msg.header
        digit_center_pose.header.frame_id = "/digit/"

        (offset_x, offset_y, offset_z) = (0.03937, 0.014, -0.007)
        if (self.obj_shape == "disc"):
            # (offset_x, offset_y, offset_z) = (0.03937, 0.011, -0.007)
            (offset_x, offset_y, offset_z) = (0.03937, 0.014, -0.007)
        elif (self.obj_shape == "rectangle"):
            (offset_x, offset_y, offset_z) = (0.03937, 0.014, -0.007)
        elif (self.obj_shape == "ellipse"):
            (offset_x, offset_y, offset_z) = (0.03937, 0.014, -0.007)
        else:
            rospy.logwarn(
                "/digit/center offset not defined for object shape {0}. Using default value.".format(self.obj_shape))

        digit_center_pose.pose.position.x = offset_x
        digit_center_pose.pose.position.y = offset_y
        digit_center_pose.pose.position.z = offset_z

        digit_center_pose.pose.orientation.x = 0
        digit_center_pose.pose.orientation.y = 0
        digit_center_pose.pose.orientation.z = 0
        digit_center_pose.pose.orientation.w = 1

        self.digit_center_pub.publish(digit_center_pose)

    def callback_object_pose(self, msg):
        object_center_pose = PoseStamped()

        object_center_pose.header = msg.header
        object_center_pose.header.frame_id = "/object/"

        (offset_x, offset_y, offset_z) = (0, 0, 0)
        (offset_ang_x, offset_ang_y, offset_ang_z) = (0, 0, 0)
        if (self.obj_shape == "disc"):
            (offset_x, offset_y, offset_z) = (0.0103, 0.0102, 0)
        elif (self.obj_shape == "rectangle"):
            (offset_x, offset_y, offset_z) = (0.0070, -0.0091, 0)
        elif (self.obj_shape == "ellipse"):
            (offset_x, offset_y, offset_z) = (0.0076, -0.0066, 0)
            (offset_ang_x, offset_ang_y, offset_ang_z) = (0, 0, 0.2139)
        else:
            rospy.logwarn(
                "/object/center offset not defined for object shape {0}. Using default value.".format(self.obj_shape))

        object_center_pose.pose.position.x = offset_x
        object_center_pose.pose.position.y = offset_y
        object_center_pose.pose.position.z = offset_z

        rot_quat = tf.transformations.quaternion_from_euler(offset_ang_x, offset_ang_y, offset_ang_z)
        object_center_pose.pose.orientation.x = rot_quat[0]
        object_center_pose.pose.orientation.y = rot_quat[1]
        object_center_pose.pose.orientation.z = rot_quat[2]
        object_center_pose.pose.orientation.w = rot_quat[3]

        self.object_center_pub.publish(object_center_pose)

    def handler_digit_center_tf(self, msg):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w),
                         msg.header.stamp, "/digit/center/", msg.header.frame_id)

    def handler_object_center_tf(self, msg):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w),
                         msg.header.stamp, "/object/center/", msg.header.frame_id)


def main():
    rospy.init_node("tf_broadcaster")
    rospy.loginfo("Initialized tf_broadcaster node.")

    tf_broadcaster = TFBroadcaster()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
