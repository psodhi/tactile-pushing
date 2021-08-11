# Copyright (c) Facebook, Inc. and its affiliates.

""" Saves mean/std no contact image to file to be used to detect contact """


import io
import os
import json
import numpy as np
import rospy
import rosbag
import rospkg
import cv2
import imageio
from PIL import Image
import skimage.draw
import matplotlib.pyplot as plt


## disc pushing datasets ##
# data_dir = "/home/paloma/Dropbox/fair_internship_20/datasets/digit_pushing/bags/disc/"
# bag_name = "20200619_disc_6in_new_firmware.bag"
# bag_name = "20200624_pushing-6in-disc-straight-line.bag"
# bag_name = "20200624_pushing-6in-disc-curves.bag"
# bag_name = "20200624_pushing-6in-disc-trial1.bag"

## rectangle pushing datasets ##
# data_dir = "/home/paloma/Dropbox/fair_internship_20/datasets/digit_pushing/bags/rectangle/"
# bag_name = "20200824_rectangle-calibration.bag"
# bag_name = "20200824_rectangle-pushing-1.bag"
# bag_name = "20200928_rectangle-pushing-edges.bag"
# bag_name = "20200928_rectangle-pushing-corners.bag"

## ellipse pushing datasets ##
data_dir = "/home/paloma/Dropbox/fair_internship_20/datasets/digit_pushing/bags/ellipse/"
bag_name = "20200928_ellipse-calibration.bag"
# bag_name = "20200928_ellipse-pushing-straight.bag"
# bag_name = "20200928_ellipse-pushing.bag"

# Time ranges (in secs) of the noncontact samples, relative to start of bag
noncontact_ranges = [(rospy.Duration(0), rospy.Duration(2))]

# tactile_topic = "/digit/digit_alpha/image_raw/compressed"
tactile_topic = "/digit/digit_alpha/image_raw"

# Convert a ROS Image message to a RGB numpy array, decompressing if needed.
def rosimg_to_numpy(imgmsg):
    if hasattr(imgmsg, 'format') and 'compressed' in imgmsg.format:
        return np.asarray(Image.open(io.BytesIO(imgmsg.data)))

    return np.frombuffer(imgmsg.data, dtype=np.uint8).reshape(imgmsg.height, imgmsg.width, 3)[:, :, ::-1]

# Returns true of msg sent during one of the specified ranges, relative to start_time.
def time_range_check(msg, start_time, ranges):
    for time_range in ranges:
        if (msg.header.stamp - start_time > time_range[0] and
                msg.header.stamp - start_time < time_range[1]):
            return True
    return False


# Extract the references (non-contact) images
bag = rosbag.Bag(data_dir + bag_name, 'r')

messages = bag.read_messages(topics=[tactile_topic])
first_msg = next(messages)
bag_start_time = first_msg.message.header.stamp

messages = bag.read_messages(topics=[tactile_topic])
images = np.array([rosimg_to_numpy(msg.message) for msg in messages])

messages = bag.read_messages(topics=[tactile_topic])
ref_images = [img for (img, msg) in zip(images, messages)
              if time_range_check(msg.message, bag_start_time, noncontact_ranges)]
ref_images = np.array(ref_images)

mean_img = ref_images.mean(axis=0)
std_img = ref_images.std(axis=0)

# save mean/std image for use later
# bag_name = "20200824_rectangle-pushing-3.bag"
save_imgs = True
if save_imgs:
    rospack = rospkg.RosPack()
    path_pkg = rospack.get_path("digit_pushing")

    bag_dir_name = bag_name.split('.')[0]
    dstdir = "{0}/local/resources/digit/{1}".format(path_pkg, bag_dir_name)
    cmd = "mkdir -p {0}".format(dstdir)
    os.popen(cmd, 'r')

    # write as .png -- doesn't preserve floating point precision
    imageio.imwrite("{0}/mean_img.png".format(dstdir), mean_img.astype(np.uint8))
    imageio.imwrite("{0}/std_img.png".format(dstdir), std_img.astype(np.uint8))

    # write as .json
    outfile = "{0}/mean_std_img.json".format(dstdir)
    data = {'mean_img': mean_img.tolist(),
            'std_img': std_img.tolist()}
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)

    print("Wrote mean, std imgs to: {0}".format(dstdir))
    
# plot saved mean/std images
plt.figure()
plt.imshow(mean_img.astype(np.uint8))
plt.title("Mean Non-Contact Image")
plt.figure()
plt.title("Non-Contact Image Noise")
plt.imshow(std_img/std_img.max())