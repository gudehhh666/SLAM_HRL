import numpy as np
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
import math
import torch
device0 = torch.device("cuda:0")

def quaternion2euler(quaternion):
    quaternion_ = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    r = R.from_quat(quaternion_)
    euler = r.as_euler('xyz', degrees=True)
    return euler


def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion


def euler2rotation(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_dcm()
    return rotation_matrix


def q2r(q):
    R = np.zeros([3, 3])
    R[0, 0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z
    R[0, 1] = 2 * q.x * q.y - 2 * q.w * q.z
    R[0, 2] = 2 * q.x * q.z + 2 * q.w * q.y
    R[1, 0] = 2 * q.x * q.y + 2 * q.w * q.z
    R[1, 1] = 1 - 2 * q.x * q.x - 2 * q.z * q.z
    R[1, 2] = 2 * q.y * q.z - 2 * q.w * q.x
    R[2, 0] = 2 * q.x * q.z - 2 * q.w * q.y
    R[2, 1] = 2 * q.y * q.z + 2 * q.w * q.x
    R[2, 2] = 1 - 2 * q.x * q.x - 2 * q.y * q.y
    return R


def q_T(orientation, position):
    rotation_quaternion = np.asarray([orientation.w, orientation.x, orientation.y, orientation.z])
    translation = np.asarray([position[0], position[1], position[2]])
    T_qua2rota = RigidTransform(rotation_quaternion, translation)
    return T_qua2rota


def pose_gt2tp(R_ini, t_ini, gt_pos):
    s_y = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])
    R_ini_ = np.dot(s_y, R_ini)
    R_ini_0 = np.dot(R_ini_, s_y)
    R_ini_1 = R_ini_0.transpose()
    a = np.array([gt_pos[0], -gt_pos[1], gt_pos[2]])
    t = np.array([t_ini[0], -t_ini[1], t_ini[2]])
    a_ = np.dot(R_ini_1, a) - np.dot(R_ini_1, t)
    a_[2] = -a_[2]
    return a_


def pose_tp2gt(R_ini, t_ini, tp_pos):
    s_y = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])
    R_ini_ = np.dot(s_y, R_ini)
    R_ini_0 = np.dot(R_ini_, s_y)
    a = np.array([tp_pos[0], tp_pos[1], -tp_pos[2]])
    t = np.array([t_ini[0], -t_ini[1], t_ini[2]])
    a_ = np.dot(R_ini_0, a) + t
    a_1 = np.array([a_[0], -a_[1], a_[2]])
    return a_1

def get_goalpoint_tp(R_ini, t_ini, cur_pos_gt, goal_pos_gt, cur_pos_tp):
    # print ('get_goalpoint_tp!!!')
    # print ('cur_pos_gt: ', cur_pos_gt)
    # print ('goal_pos_gt: ', goal_pos_gt)
    current_pos_tp = pose_gt2tp(R_ini, t_ini, cur_pos_gt)
    # print ('current_pos_tp: ', current_pos_tp)
    goal_pos_tp = pose_gt2tp(R_ini, t_ini, goal_pos_gt)
    # print ('goal_pos_tp: ', goal_pos_tp)
    dis_tp = goal_pos_tp - current_pos_tp
    # print('dis_tp', dis_tp)
    # print('cur_pos_tp', cur_pos_tp)

    goal_pos_tp = cur_pos_tp + dis_tp
    # print ('goal_pos_tp(final): ', goal_pos_tp)
    return goal_pos_tp

# def map2gt2(R_ini, t_ini, map_):  # local_goal_是栅格坐标
#     local_goal_x_tp = map_[0]*0.04-20
#     local_goal_y_tp = map_[1]*0.04-20
#     local_goal_tp = np.array([local_goal_x_tp, 0, local_goal_y_tp])
#     # print('local_goal_tp', local_goal_tp)
#     local_goal_gt = pose_tp2gt(R_ini, t_ini, local_goal_tp)
#     return local_goal_gt

def map2gt(R_ini, t_ini, map_, cur_pos_gt, cur_pos_tp):  # map_.  是栅格坐标
    local_goal_x_tp = map_[0]*0.04-20
    local_goal_y_tp = map_[1]*0.04-20
    local_goal_tp = np.array([local_goal_x_tp, 0, local_goal_y_tp])
    # print('local_goal_tp', local_goal_tp)
    local_goal_gt = pose_tp2gt(R_ini, t_ini, local_goal_tp)
    cur_pos_gt_ = pose_tp2gt(R_ini, t_ini, cur_pos_tp)
    dis_gt = local_goal_gt-cur_pos_gt_
    local_goal_gt = cur_pos_gt + dis_gt
    return local_goal_gt


def angle_GT(goal_pos_gt2, current_pos_gt, angle_tensor_gt):
    if (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[2] > current_pos_gt[2]):
        # print ('GT case 1')
        angle_goal = float((goal_pos_gt2[2] - current_pos_gt[2]) / (goal_pos_gt2[0] - current_pos_gt[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_gt: ', angle_tensor_gt)
        if (angle_tensor_gt <= 0) and (angle_tensor_gt >= -angle_goal_jiaodu):
            angle_final_gt = -angle_goal_jiaodu - angle_tensor_gt
        elif (angle_tensor_gt < -angle_goal_jiaodu):
            angle_final_gt = -angle_goal_jiaodu - angle_tensor_gt
        elif (angle_tensor_gt > 0):
            angle_final_gt = -angle_goal_jiaodu - angle_tensor_gt
            if (angle_final_gt < -180):
                angle_final_gt = angle_final_gt + 360
    elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[2] < current_pos_gt[2]):
        # print ('GT case 2')
        angle_goal = float((current_pos_gt[2] - goal_pos_gt2[2]) / (goal_pos_gt2[0] - current_pos_gt[0]))
                    # print ('angle_goal: ', angle_goal)
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_gt: ', angle_tensor_gt)
        if (angle_tensor_gt >= 0) and (angle_tensor_gt <= angle_goal_jiaodu):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
        elif (angle_tensor_gt > angle_goal_jiaodu) and (angle_tensor_gt <= 180):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
        elif (angle_tensor_gt < 0):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
            if (angle_final_gt >= 180):
                angle_final_gt = angle_final_gt - 360
    elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[2] < current_pos_gt[2]):
        # print ('GT case 3')
        angle_goal = float((current_pos_gt[2] - goal_pos_gt2[2]) / (current_pos_gt[0] - goal_pos_gt2[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = 180 -  math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_gt: ', angle_tensor_gt)
        if (angle_tensor_gt >= 0):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
        elif (angle_tensor_gt < 0):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
            if (angle_final_gt >= 180):
                angle_final_gt = angle_final_gt - 360
    elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[2] > current_pos_gt[2]):
        # print ('GT case 4')
        angle_goal = float((goal_pos_gt2[2] - current_pos_gt[2]) / (current_pos_gt[0] - goal_pos_gt2[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu) - 180
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_gt: ', angle_tensor_gt)
        if (angle_tensor_gt <= 0):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
        elif (angle_tensor_gt > 0):
            angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
            if (angle_final_gt < -180):
                angle_final_gt = angle_final_gt + 360
    elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[2] == current_pos_gt[2]):
        # print ('GT arrive!!!')
        angle_final = 0
        angle_final_gt = torch.tensor(angle_final).to(device0)
    elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[2] > current_pos_gt[2]):
        # print ('GT case 5')
        if (angle_tensor_gt > 0) and (angle_tensor_gt <= 90):
            angle_final_gt = - (angle_tensor_gt + 90)
        elif (angle_tensor_gt > 90):
            angle_final_gt = 180 - angle_tensor_gt + 90
        elif (angle_tensor_gt < 0) and (angle_tensor_gt >= -90):
            angle_final_gt = -90 - angle_tensor_gt
        elif (angle_tensor_gt < -90):
            angle_final_gt = - angle_tensor_gt - 90
    elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[2] < current_pos_gt[2]):
        # print ('GT case 6')
        if (angle_tensor_gt > -90) and (angle_tensor_gt <= 90):
            angle_final_gt = 90 - angle_tensor_gt
        elif (angle_tensor_gt > 90):
            angle_final_gt = 90 - angle_tensor_gt
        elif (angle_tensor_gt <= -90) and (angle_tensor_gt >= -180):
            angle_final_gt = -(180 + angle_tensor_gt + 90)
            # angle_final_gt = -angle_tensor_gt + 90 - 360
    elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[2] == current_pos_gt[2]):
        # print ('GT case 7')
        if (angle_tensor_gt > 0) and (angle_tensor_gt <= 180):
            angle_final_gt = - angle_tensor_gt
        elif (angle_tensor_gt > -180) and (angle_tensor_gt <= 0):
            angle_final_gt = - angle_tensor_gt
        elif (angle_tensor_gt == -180):
            angle_final_gt = angle_tensor_gt
    elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[2] == current_pos_gt[2]):
        # print ('GT case 8')
        if (angle_tensor_gt > 0) and (angle_tensor_gt <= 180):
            angle_final_gt = 180 - angle_tensor_gt
        elif (angle_tensor_gt == 0):
            angle_final_gt = -180
        elif (angle_tensor_gt < 0) and (angle_tensor_gt >= -180):
            angle_final_gt = -angle_tensor_gt - 180
    return angle_final_gt


# #没有考虑Y轴是反的
# def angle_GT(goal_pos_gt2, current_pos_gt, angle_tensor_gt):
#     if (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[2] > current_pos_gt[2]):
#         print ('GT case 1')
#         angle_goal = float((goal_pos_gt2[2] - current_pos_gt[2]) / (goal_pos_gt2[0] - current_pos_gt[0]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_gt: ', angle_tensor_gt)
#         if (angle_tensor_gt >= 0) and (angle_tensor_gt <= angle_goal_jiaodu):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#         elif (angle_tensor_gt > angle_goal_jiaodu) and (angle_tensor_gt <= 180):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#         elif (angle_tensor_gt < 0):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#             if (angle_final_gt >= 180):
#                 angle_final_gt = angle_final_gt - 360
#     elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[2] < current_pos_gt[2]):
#         print ('GT case 2')
#         angle_goal = float((goal_pos_gt2[2] - current_pos_gt[2]) / (goal_pos_gt2[0] - current_pos_gt[0]))
#                     # print ('angle_goal: ', angle_goal)
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_gt: ', angle_tensor_gt)
#         if (angle_tensor_gt <= 0) and (angle_tensor_gt >= angle_goal_jiaodu):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#         elif (angle_tensor_gt < angle_goal_jiaodu):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#         elif (angle_tensor_gt > 0):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#             if (angle_final_gt < -180):
#                 angle_final_gt = angle_final_gt + 360
#     elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[2] < current_pos_gt[2]):
#         print ('GT case 3')
#         angle_goal = float((current_pos_gt[2] - goal_pos_gt2[2]) / (current_pos_gt[0] - goal_pos_gt2[0]))
#         # print ('angle_goal: ', angle_goal)
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu) - 180
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_gt: ', angle_tensor_gt)
#         if (angle_tensor_gt <= 0):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#         elif (angle_tensor_gt > 0):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#             if (angle_final_gt < -180):
#                 angle_final_gt = angle_final_gt + 360
#     elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[2] > current_pos_gt[2]):
#         print ('GT case 4')
#         angle_goal = float((goal_pos_gt2[2] - current_pos_gt[2]) / (current_pos_gt[0] - goal_pos_gt2[0]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = 180 -  math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_gt: ', angle_tensor_gt)
#         if (angle_tensor_gt >= 0):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#         elif (angle_tensor_gt < 0):
#             angle_final_gt = angle_goal_jiaodu - angle_tensor_gt
#             if (angle_final_gt >= 180):
#                 angle_final_gt = angle_final_gt - 360
#     elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[2] == current_pos_gt[2]):
#         print ('GT arrive!!!')
#         angle_final = 0
#         # angle_final = np.dtype('int32').type(angle_final)  
#         # angle_final = torch.from_numpy(angle_final) #TypeError: expected np.ndarray (got numpy.int32)
#         angle_final_gt = torch.tensor(angle_final).to(device0)
#     elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[2] > current_pos_gt[2]):
#         print ('GT case 5')
#         if (angle_tensor_gt > -90) and (angle_tensor_gt <= 90):
#             angle_final_gt = 90 - angle_tensor_gt
#         elif (angle_tensor_gt > 90):
#             angle_final_gt = 90 - angle_tensor_gt
#         elif (angle_tensor_gt <= -90) and (angle_tensor_gt >= -180):
#             angle_final_gt = -(180 + angle_tensor_gt + 90)
#             # angle_final_gt = -angle_tensor_gt + 90 - 360
#     elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[2] < current_pos_gt[2]):
#         print ('GT case 6')
#         if (angle_tensor_gt > 0) and (angle_tensor_gt <= 90):
#             angle_final_gt = - (angle_tensor_gt + 90)
#         elif (angle_tensor_gt > 90):
#             angle_final_gt = 180 - angle_tensor_gt + 90
#         elif (angle_tensor_gt < 0) and (angle_tensor_gt >= -90):
#             angle_final_gt = -90 - angle_tensor_gt
#         elif (angle_tensor_gt < -90):
#             angle_final_gt = - angle_tensor_gt - 90
#     elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[2] == current_pos_gt[2]):
#         print ('GT case 7')
#         if (angle_tensor_gt > 0) and (angle_tensor_gt <= 180):
#             angle_final_gt = - angle_tensor_gt
#         elif (angle_tensor_gt > -180) and (angle_tensor_gt <= 0):
#             angle_final_gt = - angle_tensor_gt
#         elif (angle_tensor_gt == -180):
#             angle_final_gt = angle_tensor_gt
#     elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[2] == current_pos_gt[2]):
#         print ('GT case 8')
#         if (angle_tensor_gt > 0) and (angle_tensor_gt <= 180):
#             angle_final_gt = 180 - angle_tensor_gt
#         elif (angle_tensor_gt == 0):
#             angle_final_gt = -180
#         elif (angle_tensor_gt < 0) and (angle_tensor_gt >= -180):
#             angle_final_gt = -angle_tensor_gt - 180
#     return angle_final_gt
        

# def angle_TP(local_goal_, current_pos_tp_, angle_tensor_tp):
#     if (local_goal_[0] > current_pos_tp_[0]) and (local_goal_[1] > current_pos_tp_[1]):
#         print ('case 1')
#         angle_goal = float((local_goal_[1] - current_pos_tp_[1]) / (local_goal_[0] - current_pos_tp_[0]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         if (angle_tensor_tp < 0):
#             angle_final = -(angle_goal_jiaodu - angle_tensor_tp)
#             if (angle_final < -180):
#                 angle_final = 360 + angle_final
#         elif (angle_tensor_tp >= 0) and (angle_tensor_tp < angle_goal_jiaodu):
#             angle_final = - (angle_goal_jiaodu - angle_tensor_tp)
#         else:
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#     elif (local_goal_[0] > current_pos_tp_[0]) and (local_goal_[1] < current_pos_tp_[1]):
#         print ('case 2')
#         # angle_goal = float((current_pos_tp_[1] - local_goal_[1]) / (current_pos_tp_[0] - local_goal_[0]))
#         angle_goal = float((local_goal_[1] - current_pos_tp_[1]) / (local_goal_[0] - current_pos_tp_[0]))
#         # print ('angle_goal: ', angle_goal)
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         if (angle_tensor_tp < 0) and (angle_tensor_tp >= angle_goal_jiaodu):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#         elif (angle_tensor_tp < angle_goal_jiaodu):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#         elif (angle_tensor_tp >= 0):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#             if (angle_final >= 180):
#                 angle_final = angle_final - 360
#     elif (local_goal_[0] < current_pos_tp_[0]) and (local_goal_[1] < current_pos_tp_[1]):
#         print ('case 3')
#         # angle_goal = float((current_pos_tp_[1] - local_goal_[1]) / (current_pos_tp_[0] - local_goal_[0]))
#         angle_goal = float((current_pos_tp_[0] - local_goal_[0]) / (current_pos_tp_[1] - local_goal_[1]))
#         # print ('angle_goal: ', angle_goal)
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = -math.degrees(angle_goal_hudu) - 90
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         if (angle_tensor_tp < 0) and (angle_tensor_tp >= angle_goal_jiaodu):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#         elif (angle_tensor_tp < angle_goal_jiaodu):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#         elif (angle_tensor_tp >= 0):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#             if (angle_final >= 180):
#                 angle_final = angle_final - 360
#     elif (local_goal_[0] < current_pos_tp_[0]) and (local_goal_[1] > current_pos_tp_[1]):
#         print ('case 4')
#         angle_goal = float((local_goal_[0] - current_pos_tp_[0]) / (current_pos_tp_[1] - local_goal_[1]))
#         # print ('angle_goal: ', angle_goal)
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu) + 90
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         if (angle_tensor_tp < 0):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#             if (angle_final < -180):
#                 angle_final = 360 + angle_final
#         elif (angle_tensor_tp >= 0) and (angle_tensor_tp < angle_goal_jiaodu):
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#         else:
#             angle_final = angle_tensor_tp - angle_goal_jiaodu
#     elif (local_goal_[0] == current_pos_tp_[0]) and (local_goal_[1] == current_pos_tp_[1]):
#         print ('case 5')
#         angle_final = 0
#         # angle_final = np.dtype('int32').type(angle_final)  
#         # angle_final = torch.from_numpy(angle_final) #TypeError: expected np.ndarray (got numpy.int32)
#         angle_final = torch.tensor(angle_final).to(device0)
#     elif (local_goal_[0] == current_pos_tp_[0]) and (local_goal_[1] > current_pos_tp_[1]):
#         print ('case 6')
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         if (angle_tensor_tp <= 0) and (angle_tensor_tp >= -90):
#             angle_final = angle_tensor_tp - 90
#         elif (angle_tensor_tp < -90) and (angle_tensor_tp >= -180):
#             angle_final = 180 + angle_tensor_tp + 90
#         elif (angle_tensor_tp > 0) and (angle_tensor_tp <= 90):
#             angle_final = angle_tensor_tp - 90
#         elif (angle_tensor_tp > 90) and (angle_tensor_tp < 180):
#             angle_final = angle_tensor_tp - 90
#     elif (local_goal_[0] == current_pos_tp_[0]) and (local_goal_[1] < current_pos_tp_[1]):
#         print ('case 7')
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         if (angle_tensor_tp <= 0) and (angle_tensor_tp >= -90):
#             angle_final = 90 + angle_tensor_tp
#         elif (angle_tensor_tp < -90) and (angle_tensor_tp >= -180):
#             angle_final = 90 + angle_tensor_tp
#         elif (angle_tensor_tp > 0) and (angle_tensor_tp < 90):
#             angle_final = 90 + angle_tensor_tp
#         elif (angle_tensor_tp > 90):
#             angle_final = - (180 - angle_tensor_tp + 90)
#     elif (local_goal_[0] > current_pos_tp_[0]) and (local_goal_[1] == current_pos_tp_[1]):
#         print ('case 8')
#         print ('angle_tensor_tp: ', angle_tensor_tp)
#         angle_final = angle_tensor_tp
#     elif (local_goal_[0] < current_pos_tp_[0]) and (local_goal_[1] == current_pos_tp_[1]):
#         print ('case 9')
#         print ('angle_tensor: ', angle_tensor_tp)
#         if (angle_tensor_tp <= 0):
#             angle_final = 180 + angle_tensor_tp
#         else:
#             angle_final = angle_tensor_tp - 180
#     return angle_final

def angle_TP(goal_pos_gt2, current_pos_gt, angle_tensor_gt):
    if (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[1] > current_pos_gt[1]):
        # print ('TP case 1')
        angle_goal = float((goal_pos_gt2[1] - current_pos_gt[1]) / (goal_pos_gt2[0] - current_pos_gt[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_tp: ', angle_tensor_gt)
        if (angle_tensor_gt >= 0) and (angle_tensor_gt <= 90 - angle_goal_jiaodu):
            angle_final_gt = -(90 - angle_tensor_gt - angle_goal_jiaodu)
        elif (angle_tensor_gt > (90 - angle_goal_jiaodu)):
            angle_final_gt = angle_tensor_gt - (90 - angle_goal_jiaodu)
        elif (angle_tensor_gt < 0):
            angle_final_gt = angle_tensor_gt - (90 - angle_goal_jiaodu)
            if (angle_final_gt < -180):
                angle_final_gt = angle_final_gt + 360
    elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[1] < current_pos_gt[1]):
        # print ('TP case 2')
        angle_goal = float((current_pos_gt[1] - goal_pos_gt2[1]) / (goal_pos_gt2[0] - current_pos_gt[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_tp: ', angle_tensor_gt)
        if (angle_tensor_gt >= 0) and (angle_tensor_gt <= (90 + angle_goal_jiaodu)):
            angle_final_gt = -(90 + angle_goal_jiaodu - angle_tensor_gt)
        elif (angle_tensor_gt > (90 + angle_goal_jiaodu)):
            angle_final_gt = angle_tensor_gt - 90 - angle_goal_jiaodu
        elif (angle_tensor_gt < 0):
            angle_final_gt = angle_tensor_gt - 90 - angle_goal_jiaodu
            if (angle_final_gt < -180):
                angle_final_gt = angle_final_gt + 360
    elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[1] < current_pos_gt[1]):
        # print ('TP case 3')
        angle_goal = float((current_pos_gt[1] - goal_pos_gt2[1]) / (current_pos_gt[0] - goal_pos_gt2[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_tp: ', angle_tensor_gt)
        if (angle_tensor_gt <= 0) and (angle_tensor_gt >= (-90 - angle_goal_jiaodu)):
            angle_final_gt = 90 + angle_goal_jiaodu + angle_tensor_gt
        elif (angle_tensor_gt < (-90 - angle_goal_jiaodu)):
            angle_final_gt = angle_tensor_gt + 90 + angle_goal_jiaodu
        elif (angle_tensor_gt > 0):
            angle_final_gt = 90 + angle_goal_jiaodu + angle_tensor_gt
            if (angle_final_gt >= 180):
                angle_final_gt = angle_final_gt - 360
    elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[1] > current_pos_gt[1]):
        # print ('TP case 4')
        angle_goal = float((goal_pos_gt2[1] - current_pos_gt[1]) / (current_pos_gt[0] - goal_pos_gt2[0]))
        angle_goal_hudu = math.atan(angle_goal)
        angle_goal_jiaodu = math.degrees(angle_goal_hudu)
        # print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
        # print ('angle_tensor_gt: ', angle_tensor_gt)
        if (angle_tensor_gt <= 0) and (angle_tensor_gt >= (angle_goal_jiaodu - 90)):
            angle_final_gt = 90 - angle_goal_jiaodu + angle_tensor_gt
        elif (angle_tensor_gt < (angle_goal_jiaodu - 90)):
            angle_final_gt = angle_tensor_gt + 90 - angle_goal_jiaodu
        elif (angle_tensor_gt > 0):
            angle_final_gt = angle_tensor_gt + 90 - angle_goal_jiaodu
            if (angle_final_gt >= 180):
                angle_final_gt = angle_final_gt - 360
    elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[1] == current_pos_gt[1]):
        # print ('TP arrive!!!')
        angle_final = 0
        # angle_final = np.dtype('int32').type(angle_final)  
        # angle_final = torch.from_numpy(angle_final) #TypeError: expected np.ndarray (got numpy.int32)
        angle_final_gt = torch.tensor(angle_final).to(device0)
    elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[1] > current_pos_gt[1]):
        # print ('TP case 5')
        angle_final_gt = angle_tensor_gt
    elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[1] < current_pos_gt[1]):
        # print ('TP case 6')
        if (angle_tensor_gt >= 0) and (angle_tensor_gt < 180):
            angle_final_gt = -(180 - angle_tensor_gt)
        elif (angle_tensor_gt >= -180) and (angle_tensor_gt < 0):
            angle_final_gt = 180 + angle_tensor_gt
    elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[1] == current_pos_gt[1]):
        # print ('TP case 7')
        if (angle_tensor_gt >= 0) and (angle_tensor_gt < 90):
            angle_final_gt = angle_tensor_gt - 90
        elif (angle_tensor_gt >= 90) and (angle_tensor_gt < 180):
            angle_final_gt = angle_tensor_gt - 90
        elif (angle_tensor_gt < 0):
            angle_final_gt = angle_tensor_gt - 90
            if (angle_final_gt < -180):
                angle_final_gt = 360 + angle_final_gt
    elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[1] == current_pos_gt[1]):
        # print ('TP case 8')
        if (angle_tensor_gt < 0) and (angle_tensor_gt >= -90):
            angle_final_gt = angle_tensor_gt + 90
        elif (angle_tensor_gt < -90):
            angle_final_gt = angle_tensor_gt + 90
        elif (angle_tensor_gt >= 0):
            angle_final_gt = angle_tensor_gt + 90
            if (angle_final_gt >= 180):
                angle_final_gt = angle_final_gt - 360
    return angle_final_gt


# #第一个元素为Y轴，第二个元素为X轴
# def angle_TP(goal_pos_gt2, current_pos_gt, angle_tensor_gt):
#     if (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[1] > current_pos_gt[1]):
#         print ('TP case 1')
#         angle_goal = float((goal_pos_gt2[0] - current_pos_gt[0]) / (goal_pos_gt2[1] - current_pos_gt[1]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_gt)
#         if (angle_tensor_gt >= 0) and (angle_tensor_gt <= 90 - angle_goal_jiaodu):
#             angle_final_gt = -(90 - angle_tensor_gt - angle_goal_jiaodu)
#         elif (angle_tensor_gt > (90 - angle_goal_jiaodu)):
#             angle_final_gt = angle_tensor_gt - (90 - angle_goal_jiaodu)
#         elif (angle_tensor_gt < 0):
#             angle_final_gt = angle_tensor_gt - (90 - angle_goal_jiaodu)
#             if (angle_final_gt < -180):
#                 angle_final_gt = angle_final_gt + 360
#     elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[1] < current_pos_gt[1]):
#         print ('TP case 2')
#         angle_goal = float((goal_pos_gt2[0] - current_pos_gt[0]) / (current_pos_gt[1] - goal_pos_gt2[1]))
#         # angle_goal = float((current_pos_gt[1] - goal_pos_gt2[1]) / (goal_pos_gt2[0] - current_pos_gt[0]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_gt)
#         if (angle_tensor_gt <= 0) and (angle_tensor_gt >= (angle_goal_jiaodu - 90)):
#             angle_final_gt = 90 - angle_goal_jiaodu + angle_tensor_gt
#         elif (angle_tensor_gt < (angle_goal_jiaodu - 90)):
#             angle_final_gt = angle_tensor_gt + 90 - angle_goal_jiaodu
#         elif (angle_tensor_gt > 0):
#             angle_final_gt = angle_tensor_gt + 90 - angle_goal_jiaodu
#             if (angle_final_gt >= 180):
#                 angle_final_gt = angle_final_gt - 360
#     elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[1] < current_pos_gt[1]):
#         print ('TP case 3')
#         angle_goal = float((current_pos_gt[0] - goal_pos_gt2[0]) / (current_pos_gt[1] - goal_pos_gt2[1]))
#         # angle_goal = float((current_pos_gt[1] - goal_pos_gt2[1]) / (current_pos_gt[0] - goal_pos_gt2[0]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_tp: ', angle_tensor_gt)
#         if (angle_tensor_gt <= 0) and (angle_tensor_gt >= (-90 - angle_goal_jiaodu)):
#             angle_final_gt = 90 + angle_goal_jiaodu + angle_tensor_gt
#         elif (angle_tensor_gt < (-90 - angle_goal_jiaodu)):
#             angle_final_gt = angle_tensor_gt + 90 + angle_goal_jiaodu
#         elif (angle_tensor_gt > 0):
#             angle_final_gt = 90 + angle_goal_jiaodu + angle_tensor_gt
#             if (angle_final_gt >= 180):
#                 angle_final_gt = angle_final_gt - 360
#     elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[1] > current_pos_gt[1]):
#         print ('TP case 4')
#         angle_goal = float((current_pos_gt[0] - goal_pos_gt2[0]) / (goal_pos_gt2[1] - current_pos_gt[1]))
#         # angle_goal = float((goal_pos_gt2[1] - current_pos_gt[1]) / (current_pos_gt[0] - goal_pos_gt2[0]))
#         angle_goal_hudu = math.atan(angle_goal)
#         angle_goal_jiaodu = math.degrees(angle_goal_hudu)
#         print ('angle_goal_jiaodu: ', angle_goal_jiaodu)
#         print ('angle_tensor_gt: ', angle_tensor_gt)
#         if (angle_tensor_gt >= 0) and (angle_tensor_gt <= (90 + angle_goal_jiaodu)):
#             angle_final_gt = -(90 + angle_goal_jiaodu - angle_tensor_gt)
#         elif (angle_tensor_gt > (90 + angle_goal_jiaodu)):
#             angle_final_gt = angle_tensor_gt - 90 - angle_goal_jiaodu
#         elif (angle_tensor_gt < 0):
#             angle_final_gt = angle_tensor_gt - 90 - angle_goal_jiaodu
#             if (angle_final_gt < -180):
#                 angle_final_gt = angle_final_gt + 360
#     elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[1] == current_pos_gt[1]):
#         print ('TP arrive!!!')
#     elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[1] > current_pos_gt[1]):
#         print ('TP case 5')
#         angle_final_gt = - angle_tensor_gt
#     elif (goal_pos_gt2[0] == current_pos_gt[0]) and (goal_pos_gt2[1] < current_pos_gt[1]):
#         print ('TP case 6')
#         if (angle_tensor_gt >= 0) and (angle_tensor_gt < 180):
#             angle_final_gt = 180 - angle_tensor_gt
#         elif (angle_tensor_gt >= -180) and (angle_tensor_gt < 0):
#             angle_final_gt = - (180 + angle_tensor_gt)
#     elif (goal_pos_gt2[0] > current_pos_gt[0]) and (goal_pos_gt2[1] == current_pos_gt[1]):
#         print ('TP case 7')
#         if (angle_tensor_gt >= 0) and (angle_tensor_gt < 90):
#             angle_final_gt = 90 - angle_tensor_gt
#         elif (angle_tensor_gt >= 90) and (angle_tensor_gt < 180):
#             angle_final_gt = 90 - angle_tensor_gt
#         elif (angle_tensor_gt < 0):
#             angle_final_gt = - (90 + 180 + angle_tensor_gt)
#             if (angle_final_gt < -180):
#                 angle_final_gt = 360 + angle_final_gt
#     elif (goal_pos_gt2[0] < current_pos_gt[0]) and (goal_pos_gt2[1] == current_pos_gt[1]):
#         print ('TP case 8')
#         if (angle_tensor_gt < 0) and (angle_tensor_gt >= -90):
#             angle_final_gt = - (angle_tensor_gt + 90)
#         elif (angle_tensor_gt < -90):
#             angle_final_gt = - angle_tensor_gt - 90
#         elif (angle_tensor_gt >= 0):
#             angle_final_gt = - angle_tensor_gt - 90
#             if (angle_final_gt < -180):
#                 angle_final_gt = 360 + angle_final_gt
#     return angle_final_gt





if __name__ == "__main__":
    test = np.random.random([1, 2])
    print('test', test)
    a = test[0][0]
    b= test[0][1]
    print(a, b)