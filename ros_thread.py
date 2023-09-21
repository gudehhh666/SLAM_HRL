import sys

# sys.path.append('/home/jin/SLAM/ORB_SLAM_rgbd_loop_0410/ORB_SLAM_rgbd_loop/Examples/ROS/devel/lib/python2.7/dist-packages/')


sys.path.append('/media/vision/data/wang/ORB_SLAM_rgbd_loop_0410/ORB_SLAM_rgbd_loop/Examples/ROS/devel/lib/python2.7/dist-packages/')


sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import threading
import rospy
import glovar
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray
from copy import deepcopy
from orb_slam2_ros.msg import Sts_mpt

threadLock = threading.Lock()


class rosthread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("=================================开启线程： " + self.name)
        TestMap()
        rospy.spin()


class TestMap(object):
    def __init__(self):
        # global map_data_
        # global camera_trajectory
        self.map_data = []
        self.map_height = 0
        self.map_width = 0
        self.came_pose = {}
        self.cam_ori = {}
        # Give the node a name
        # rospy.init_node('test_map', anonymous=False, disable_signals=True)
        rospy.Subscriber("/map", OccupancyGrid, self.get_map_data, queue_size=1)
        rospy.Subscriber("/trajectory", Float32MultiArray, self.get_trajectory, queue_size=1)
        rospy.Subscriber("/RGBD/Sts_mpt", Sts_mpt, self.get_sts_mpt, queue_size=1)

    def get_map_data(self, msg):
        self.map_height = msg.info.height  # pgm 图片属性的像素值（栅格地图初始化大小——高度上的珊格个数）
        self.map_width = msg.info.width  # pgm 图片属性的像素值（栅格地图初始化大小中的宽——宽度上的珊格个数）
        origin_x = msg.info.origin.position.x  # ROS 建图的 origin
        origin_y = msg.info.origin.position.y  # ROS 建图的 origin
        resolution = msg.info.resolution  # ROS 建图的分辨率 resolution（栅格地图分辨率对应栅格地图中一小格的长和宽）
        data_size = len(msg.data)
        # print(data_size)
        glovar.map_data_['map_data'] = deepcopy(msg.data)
        # print ('test ros_thread_map_data', type(glovar.map_data_['map_data']))

    def get_trajectory(self, msg):
        glovar.camera_trajectory['data'] = deepcopy(msg.data)
        # print ('test ros_thread_camera_trajectory', type(glovar.camera_trajectory['data']))

    def get_sts_mpt(self, msg):
        glovar.orb_sts_mpt['mpt'] = deepcopy(msg.mpt)
        glovar.orb_sts_mpt['sts'] = deepcopy(msg.sts)
        # print('orb_sts:', glovar.orb_sts_mpt['sts'])
        # orb_sts_mpt = glovar.orb_sts_mpt['mpt'].reshape(640, 480)
        # print(orb_sts_mpt)
        # print('orb_sts:', glovar.orb_sts_mpt['sts'])


# global map_data_
# global camera_trajectory


