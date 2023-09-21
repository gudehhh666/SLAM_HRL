import numpy as np
import torch
import cv2
from torch.nn import functional as F
from habitat_baselines.slambased.path_planners import DifferentiableStarPlanner
from habitat_baselines.slambased.utils import generate_2dgrid
from habitat_baselines.slambased.reprojection import (homogenize_p, planned_path2tps, get_distance, get_direction,
                                                      project_tps_into_worldmap)
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

#device = torch.device("cuda:0")
device = torch.device("cpu")

planner = DifferentiableStarPlanner(
    max_steps=500,
    preprocess=True,
    beta=100,
    device=device,
)


# planner = DifferentiableStarPlanner(
#     max_steps=500,
#     preprocess=True,
#     beta=100,
#     device=device,
# )


def get_current_pose_tp(camera_trajectory):  #将当前位置转换到tp图上
    # print ('len(camera_trajectory[data]):', len(camera_trajectory['data']))
    if len(camera_trajectory['data']) > 0:
        predicted_pose = np.array(camera_trajectory['data'][-12:])
        pose6D = homogenize_p(torch.from_numpy(predicted_pose).view(3, 4).to(device)).view(1, 4, 4)
        # print('pose6d', pose6D)
        current_pos = project_tps_into_worldmap(pose6D.view(1, 4, 4), 0.04, 40, True)
    else:
        current_pos = torch.zeros([1, 2])
    return current_pos


def get_current_pose(camera_trajectory):
    # print ('get_current_pose!!!')
    if len(camera_trajectory['data']) > 0:
        predicted_pose = np.array(camera_trajectory['data'][-12:])
        current_pos_t = [predicted_pose[3], predicted_pose[7], predicted_pose[11]]
        current_pos_r = np.array([[predicted_pose[0], predicted_pose[1], predicted_pose[2]],
                                  [predicted_pose[4], predicted_pose[5], predicted_pose[6]],
                                  [predicted_pose[8], predicted_pose[9], predicted_pose[10]]])
    else:
        current_pos_r = []
        current_pos_t = []
    # print ('current_pos_r: ', current_pos_r)
    # print ('current_pos_t: ', current_pos_t)
    return current_pos_r, current_pos_t


def get_mp_tp(map_data):
    map_data_arr = np.array(map_data['map_data']).astype(np.int16)
    return map_data_arr


def mp_tp_trans(map_data_arr):
    if len(map_data_arr) > 0:
        map_data_arr_reshape = map_data_arr.reshape((1000, 1000))
        map_data_arr_reshape = np.add(map_data_arr_reshape, 1)
        map_data_arr_reshape = np.clip(map_data_arr_reshape, 0, 2).astype(np.uint8)
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        map_data_arr_reshape_numpy = recolor_map[map_data_arr_reshape]
        return map_data_arr_reshape_numpy
    else:
        print('map_data is blank')
    # map_data_arr_reshape_torch = torch.from_numpy(map_data_arr_reshape)
    # map_data_arr_reshape_torch_unsqueeze = map_data_arr_reshape_torch.unsqueeze(0).unsqueeze(0).float().to(device)





# def plan_path(map_data_arr, current_pos, goal_pos, height=1000, width=1000):
#     if len(map_data_arr) > 0:
#         map_data_arr_reshape = map_data_arr.reshape((1000, 1000))
#         map_data_arr_reshape = np.add(map_data_arr_reshape, 1)
#         map_data_arr_reshape_tensor = torch.from_numpy(map_data_arr_reshape).to(device)
#         map_data_arr_reshape_tensor = torch.clip(map_data_arr_reshape_tensor, 0, 2)
#         map_data_arr_reshape_torch_unsqueeze = map_data_arr_reshape_tensor.unsqueeze(0).unsqueeze(
#             0).float().to(device)
#         # print(current_pos)
#         start_map = torch.zeros_like(map_data_arr_reshape_torch_unsqueeze).to(device)
#         start_map[0, 0, current_pos[0, 0].long(), current_pos[0, 1].long()] = 1.0
#         goal_map = torch.zeros_like(map_data_arr_reshape_torch_unsqueeze).to(device)
#         goal_map[0, 0, goal_pos[0, 0].long(), goal_pos[0, 1].long()] = 1.0
#         # map1 = (map_data_arr_reshape_torch_unsqueeze / float(320)) ** 2   #为什么要除320？
#         # print ('map_data_arr_reshape_torch_unsqueeze: ', map_data_arr_reshape_torch_unsqueeze.shape) #torch.Size([1, 1, 1000, 1000])
#         # for i in range(500, 570):
#         #     print ('map_data_arr_reshape_torch_unsqueeze: ', map_data_arr_reshape_torch_unsqueeze[0][0][i][470:530])
#         for i in range(0, 1000):
#             for j in range(0, 1000):
#                 if (map_data_arr_reshape_torch_unsqueeze[0][0][i][j] > 1.0):
#                     map_data_arr_reshape_torch_unsqueeze[0][0][i][j] = 600.0



#         # print ('map_data_arr_reshape_torch_unsqueeze: ', map_data_arr_reshape_torch_unsqueeze[0][0][500][480:520])
#         # print ('map_data_arr_reshape_torch_unsqueeze: ', map_data_arr_reshape_torch_unsqueeze[0][0][520][460:540])
#         # print ('map_data_arr_reshape_torch_unsqueeze: ', map_data_arr_reshape_torch_unsqueeze[480:520][480:520])
#         map1 = (map_data_arr_reshape_torch_unsqueeze / float(256)) ** 2   
#         # map1 = (map_data_arr_reshape_torch_unsqueeze / float(1)) ** 2  
#         # map1 = map1.to(device)
#         map1 = (torch.clamp(map1.to(device), min=0, max=1.0) - start_map - F.max_pool2d(goal_map, 3, stride=1,
#                                                                                         padding=1))
#         map1 = torch.relu(map1)
#         coordinatesGrid = generate_2dgrid(height, width, False)
#         if hasattr(torch.cuda, 'empty_cache'):
#             torch.cuda.empty_cache()
#         path, cost = planner(map1.to(device), coordinatesGrid.to(device), goal_map.to(device),
#                              start_map.to(device))
#         planned_waypoints = planned_path2tps(path, 0.04, 40, 0.4, False).to(device)
#         return path, cost, planned_waypoints


def plan_path(map_data_arr, current_pos, goal_pos, height=1000, width=1000):
    if len(map_data_arr) > 0:
        map_data_arr_reshape1 = map_data_arr.reshape((1000, 1000))
        #case 1
        map_data_arr_reshape_15 = np.transpose(map_data_arr_reshape1)
    
        # map_data_arr_reshape2 = np.add(map_data_arr_reshape_15, 1)
        map_data_arr_reshape_tensor1 = torch.from_numpy(map_data_arr_reshape_15).to(device).contiguous()
        map_data_arr_reshape_tensor2 = torch.clip(map_data_arr_reshape_tensor1, 0, 1)

        #膨胀
        for k in range(4):
            map_data_arr_reshape_tensor2_up = deepcopy(map_data_arr_reshape_tensor2)
            map_data_arr_reshape_tensor2_up[999][:] = map_data_arr_reshape_tensor2[0][:]
            for i in range(1 , 1000, 1):
                map_data_arr_reshape_tensor2_up[i-1][:] = map_data_arr_reshape_tensor2[i][:]
            map_data_arr_reshape_tensor2_below = deepcopy(map_data_arr_reshape_tensor2)
            map_data_arr_reshape_tensor2_below[0][:] = map_data_arr_reshape_tensor2[999][:]
            for i in range(0 ,999 ,1):
                map_data_arr_reshape_tensor2_below[i+1][:] = map_data_arr_reshape_tensor2[i][:]
            map_data_arr_reshape_tensor2_left = deepcopy(map_data_arr_reshape_tensor2)
            map_data_arr_reshape_tensor2_left[:,999] = map_data_arr_reshape_tensor2[:,0]
            for i in range(1 ,1000 ,1):
                map_data_arr_reshape_tensor2_left[:,i-1] = map_data_arr_reshape_tensor2[:,i]
            map_data_arr_reshape_tensor2_right = deepcopy(map_data_arr_reshape_tensor2)
            map_data_arr_reshape_tensor2_right[:,0] = map_data_arr_reshape_tensor2[:,999]
            for i in range(0 ,999 ,1):
                map_data_arr_reshape_tensor2_right[:,i+1] = map_data_arr_reshape_tensor2[:,i]
            map_data_arr_reshape_tensor2 = map_data_arr_reshape_tensor2 + map_data_arr_reshape_tensor2_up + map_data_arr_reshape_tensor2_below + map_data_arr_reshape_tensor2_left + map_data_arr_reshape_tensor2_right
            map_data_arr_reshape_tensor2 = torch.clip(map_data_arr_reshape_tensor2, 0, 1)


        # map_data_arr_reshape_tensor2_up = deepcopy(map_data_arr_reshape_tensor2)
        # map_data_arr_reshape_tensor2_up[999][:] = map_data_arr_reshape_tensor2[0][:]
        # for i in range(1 , 1000, 1):
        #     map_data_arr_reshape_tensor2_up[i-1][:] = map_data_arr_reshape_tensor2[i][:]
        # map_data_arr_reshape_tensor2_below = deepcopy(map_data_arr_reshape_tensor2)
        # map_data_arr_reshape_tensor2_below[0][:] = map_data_arr_reshape_tensor2[999][:]
        # for i in range(0 ,999 ,1):
        #     map_data_arr_reshape_tensor2_below[i+1][:] = map_data_arr_reshape_tensor2[i][:]
        # map_data_arr_reshape_tensor2_left = deepcopy(map_data_arr_reshape_tensor2)
        # map_data_arr_reshape_tensor2_left[:,999] = map_data_arr_reshape_tensor2[:,0]
        # for i in range(1 ,1000 ,1):
        #     map_data_arr_reshape_tensor2_left[:,i-1] = map_data_arr_reshape_tensor2[:,i]
        # map_data_arr_reshape_tensor2_right = deepcopy(map_data_arr_reshape_tensor2)
        # map_data_arr_reshape_tensor2_right[:,0] = map_data_arr_reshape_tensor2[:,999]
        # for i in range(0 ,999 ,1):
        #     map_data_arr_reshape_tensor2_right[:,i+1] = map_data_arr_reshape_tensor2[:,i]
        # map_data_arr_reshape_tensor2_final = map_data_arr_reshape_tensor2 + map_data_arr_reshape_tensor2_up + map_data_arr_reshape_tensor2_below + map_data_arr_reshape_tensor2_left + map_data_arr_reshape_tensor2_right
        # map_data_arr_reshape_tensor2_final = torch.clip(map_data_arr_reshape_tensor2_final, 0, 1)


        # map_data_arr_reshape_torch_unsqueeze = map_data_arr_reshape_tensor2_final.unsqueeze(0).unsqueeze(
        #     0).float().to(device)
        map_data_arr_reshape_torch_unsqueeze = map_data_arr_reshape_tensor2.unsqueeze(0).unsqueeze(
            0).float().to(device)

        start_map = torch.zeros_like(map_data_arr_reshape_torch_unsqueeze).to(device)
        start_map[0, 0, current_pos[0, 0].long(), current_pos[0, 1].long()] = 1.0
        goal_map = torch.zeros_like(map_data_arr_reshape_torch_unsqueeze).to(device)
        goal_map[0, 0, goal_pos[0, 0].long(), goal_pos[0, 1].long()] = 1.0
        # #补充
        # for i in range(0, 1000):
        #     for j in range(0, 1000):
        #         if (map_data_arr_reshape_torch_unsqueeze[0][0][i][j] > 1.0):
        #             map_data_arr_reshape_torch_unsqueeze[0][0][i][j] = 600.0
        # #=======
        map1 = map_data_arr_reshape_torch_unsqueeze
        # map1 = (map_data_arr_reshape_torch_unsqueeze / float(2)) ** 2   #320
        # map1 = map1.to(device)
        map1 = (torch.clamp(map1.to(device), min=0, max=1.0) - start_map - F.max_pool2d(goal_map, 3, stride=1,
                                                                                        padding=1))
        map1 = torch.relu(map1)
        coordinatesGrid = generate_2dgrid(height, width, False)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        path, cost = planner(map1.to(device), coordinatesGrid.to(device), goal_map.to(device),
                             start_map.to(device))
        planned_waypoints = planned_path2tps(path, 0.04, 40, 1.5, False).to(device)
        return path, cost, planned_waypoints



def get_local_goal_point(path, current_pos_tp):
    current_pos_tp = current_pos_tp.detach().to(device).numpy()
    # current_pos_tp = current_pos_tp.detach().cpu().numpy()
    current_pos_tp_ = np.array([current_pos_tp[0][0], current_pos_tp[0][1]])
    # print('current_pos_tp_', current_pos_tp_)
    for point in path:
        point_ = point.detach().to(device).numpy()
        # point_ = point.detach().cpu().numpy()
        point2 = point_ - current_pos_tp_
        if (point2[0] ** 2 + point2[1] ** 2) >= 100:
            return point_
    # print('goal_point in local_map')
    return path[-1].detach().to(device).numpy()
    # return path[-1].detach().cpu().numpy()


# def get_angle(R_Matrix):
#     r = R.from_matrix(R_Matrix)
#     current_pose_euro = r.as_euler('xyz', degrees=True)
#     if (current_pose_euro[0] <= 90) and (current_pose_euro[1] >= 0):
#         angle = current_pose_euro[1]
#     elif (current_pose_euro[0] > 90) and (current_pose_euro[1] > 0):
#         angle = 180 - current_pose_euro[1]
#     elif (current_pose_euro[0] >= 90) and (current_pose_euro[1] <= 0):
#         angle = -180 - current_pose_euro[1]
#     else:
#         angle = current_pose_euro[1]
#     return angle

def get_angle(R_Matrix):
    r = R.from_matrix(R_Matrix)
    current_pose_euro = r.as_euler('xyz', degrees=True)
    # print('current_pose_euro', current_pose_euro)
    if (current_pose_euro[1] <= 0) and (abs(current_pose_euro[2]) <= 90):
        angle = -current_pose_euro[1]
    elif (current_pose_euro[1] <= 0) and (abs(current_pose_euro[2]) > 90):
        angle = current_pose_euro[1] + 180
    elif (current_pose_euro[1] > 0) and (abs(current_pose_euro[2]) > 90):
        angle = current_pose_euro[1] - 180
    else:
        angle = -current_pose_euro[1]
    return angle



# def get_angle_gt(rotation_quat):
#     r = R.from_quat(rotation_quat)
#     current_pose_euro = r.as_euler('xyz', degrees=True)
#     # print ('current_pose_euro: ', current_pose_euro)
#     if (current_pose_euro[0] <= 90) and (current_pose_euro[1] >= 0):
#         angle = current_pose_euro[1]
#     elif (current_pose_euro[0] > 90) and (current_pose_euro[1] > 0):
#         angle = 180 - current_pose_euro[1]
#     elif (current_pose_euro[0] >= 90) and (current_pose_euro[1] <= 0):
#         angle = -180 - current_pose_euro[1]
#     else:
#         angle = current_pose_euro[1]
#     return angle, current_pose_euro


def get_angle_gt(rotation_quat):
    r = R.from_quat(rotation_quat)
    current_pose_euro = r.as_euler('xyz', degrees=True)
    # print(['current_pose_euro', current_pose_euro])   #改一改！！！

    if (current_pose_euro[0] < 90) and (current_pose_euro[1] > 0):
        angle = -90 + current_pose_euro[1]
    elif (current_pose_euro[0] < 90) and (current_pose_euro[1] < 0):
        angle = -90 + current_pose_euro[1]
    elif (current_pose_euro[0] >= 90) and (current_pose_euro[1] < 0):
        angle = 90 - current_pose_euro[1]
    else:
        angle = 90 - current_pose_euro[1]
    return angle




def distance(ax1, ax2):
    dis = ax1-ax2
    dis_abs = (dis[0]**2+dis[1]**2+dis[2]**2)**0.5
    return dis_abs