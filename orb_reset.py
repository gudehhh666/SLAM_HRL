import rospy
import sys
import glovar
import numpy as np

sys.path.append('/media/vision/data/wang/ORB_SLAM_rgbd_loop_0410/ORB_SLAM_rgbd_loop/Examples/ROS/devel/lib/python2.7/dist-packages/')
from orb_slam2_ros.srv import *


def orb_slam_reset():
    rospy.wait_for_service('orb_slam_reset')
    try:
        orb_ret = rospy.ServiceProxy('orb_slam_reset', orb_reset)
        sucess = orb_ret(1)
        print('success', sucess)
        return sucess
    except rospy.ServiceException as e:
        print(e)


def get_orb_sts_mpt(orb_sts_mpt):
    orb_sts = orb_sts_mpt['sts']
    if orb_sts == 2:
        orb_mpt = np.array([orb_sts_mpt['mpt']])
        orb_mpt = orb_mpt.reshape(480, 640)
        #orb_mpt = orb_mpt.reshape(512, 512)
        return orb_mpt, orb_sts
    else:
        orb_mpt = None
        print('orb_mpt is none.......')
    return orb_mpt, orb_sts


def set_glovar_var_ini():
    glovar.camera_trajectory = {'data': []}
    glovar.map_data_ = {'map_data': []}
    glovar.orb_sts_mpt = {'mpt': [], 'sts': 0}
    glovar.image = np.zeros((480, 640))
    #glovar.image = np.zeros((512, 512, 4))
    glovar.depth = np.zeros((480, 640, 4))
    #glovar.depth = np.zeros((512, 512))
    glovar.local_map = []
    glovar.local_goal_point = []
    glovar.angle = None
    # glovar.obv = {'orb_sts_mpt': glovar.orb_sts_mpt, 'image': glovar.image, 'depth': glovar.depth, 'local_map': glovar.local_map,
    #               'angle': glovar.angle}






