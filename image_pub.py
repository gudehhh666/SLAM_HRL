import sys


sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import threading
import rospy
from sensor_msgs.msg import Image as ros_Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import glovar
from PIL import Image


class image_pub(threading.Thread):
    def __init__(self, threadID, name, counter, sim):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.sim = sim

        rospy.init_node('test_pub_map', anonymous=False, disable_signals=True)
        self.resized_rgb_img = rospy.Publisher('camera/rgb/image_color', ros_Image, queue_size=10)
        self.resized_depth_img = rospy.Publisher('camera/depth/image', ros_Image, queue_size=10)

    def run(self):
        print("开启线程： " + self.name)
        bridge = CvBridge()
        while not rospy.is_shutdown():
            # threadLock2.acquire()
            obsv = self.sim.get_sensor_observations()
            image = obsv["color_sensor"]
            depth = obsv["depth_sensor"]
            # current_time = time.time()
            rgb_img = Image.fromarray(image.astype(np.uint8), mode="RGBA")
            rgb_img = rgb_img.convert('RGB')
            rgb_img_arr = np.array(rgb_img)
            cv_img = bridge.cv2_to_imgmsg(rgb_img_arr, "passthrough")
            self.resized_rgb_img.publish(cv_img)
            depth_int = (depth * 5000).astype(np.uint16)
            depth_img_arr = np.array(depth_int)
            cv_img_depth = bridge.cv2_to_imgmsg(depth_img_arr, "passthrough")
            self.resized_depth_img.publish(cv_img_depth)
            if glovar.sence_change == 1:
                # xxx.public(xxxx)     # ORB复位命令
                break                     #
            time.sleep(0.033)
