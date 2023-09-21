import numpy as np
from habitat.utils.visualizations import maps
import cv2
import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import time
import torch
import glovar
import os

output_path = "/media/vision/data/wang/codes/TD3_HRL/orbslam_sim/picture"
output_path2 = "/media/vision/data/wang/codes/PPO/map_gt/nav_output2/"
# output_path = "/home/jin/RL-code/orbslam_sim/picture"
# output_path2 = "map_gt/nav_output2/"
global num_
num_ = 1
device1 = torch.device("cpu")

def save_map(topdown_map, key_points=None):
    global num_
    if key_points is not None:
        for point in key_points:
            cv2.drawMarker(topdown_map, (int(point[0]), int(point[1])), color=(0, 255, 0), markerType=0, thickness=2)
        if topdown_map is not None:
            cv2.imwrite(os.path.join(output_path, "cv2_Habitat-Lab_maps") + str(num) + ".png",
                        topdown_map)
            num_ += 1
        else:
            print('map is blank')


def display_map(topdown_map, figure_num=1, key_points=None):
    t1 = time.time()
    global num_
    plt.figure(num=figure_num, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=3, alpha=0.8)
    # plt.savefig(os.path.join(output_path, str(count))+".png")
    # plt.savefig(os.path.join(output_path, "GT_map_with_pos")+str(num)+".png")
    num_ += 1
    # plt.show()
    plt.draw()
    plt.pause(0.5)
    t2 = time.time()
    # print("time_sapn2", t2 - t1)


def display_map2(topdown_map, t, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    # print ('33')
    plt.imshow(topdown_map)
    # print ('44')
    # plot points on map
    if key_points is not None:
        i = 0
        for point in key_points:
            if (i == 0):
                plt.plot(point[0], point[1], marker=".", markersize=10, alpha=0.8)
            elif (i == (len(key_points) - 1)):
                plt.plot(point[0], point[1], marker="^", markersize=10, alpha=0.8)
            else:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
            i = i + 1
    output_path2 = "/media/vision/data/wang/codes/TD3_HRL/safe_mode6/map_gt/date8_29_1_/" + str(glovar.test) + "/" + str(glovar.train_num) + '_' + str(glovar.scene_id[7:-4]) + '_' + str(glovar.episode_id) + '/'
    if  not os.path.exists(output_path2):
        os.makedirs(output_path2)
    # print ('55')
    save_path = str(t) + '.png'
    plt.savefig(os.path.join(output_path2, save_path))
    # print ('success save!')
    plt.show(block=False)




def convert_points_to_topdown(pathfinder, points, meters_per_pixel=0.04):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def convert_point_to_tpmap(points, meters_per_pixel=0.04):
    points_topdown = []

    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] + 20) / meters_per_pixel
        py = (point[2] + 20) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def get_hablab_topdown_map(sim_pathfinder, agent, meters_per_pixel=0.04):
    agent_state = agent.state
    agent_pos = agent_state.position
    map_height = agent_pos[1]
    hablab_topdown_map = maps.get_topdown_map(sim_pathfinder, map_height, meters_per_pixel=meters_per_pixel)
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    hablab_topdown_map = recolor_map[hablab_topdown_map]
    return hablab_topdown_map


def display_map_gt(sim_pathfinder, map_gt, agent, meters_per_pixel=0.04, fig_num=1):
    agent_pos = agent.state.position
    vis_points = [agent_pos]
    print('vis_points', vis_points)
    xy_vis_points = convert_points_to_topdown(sim_pathfinder, vis_points, meters_per_pixel)
    # print('xy_vis_points', xy_vis_points)
    display_map(map_gt, fig_num, key_points=xy_vis_points)


def display_map_tp(map_tp, current_pose):
    current_pose_trans = convert_point_to_tpmap([current_pose])
    display_map(map_tp, 2, current_pose_trans)


def display_map_tp_with_global_path(map_tp, current_pose_trans, path):
    x = []
    y = []
    x_ = []
    for iterm in path:
        pos = iterm.to(device1).numpy()
        # pos = iterm.cpu().numpy()
        x.append(pos[0])
        y.append(pos[1])

    for it in x:
        x_.append(1000.0 - it)
    print('x', x)
    print('y', y)
    t1 = time.time()
    plt.clf()
    plt.figure(num=3, figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    ax.axis("off")
    plt.imshow(map_tp)
    ax.plot(y, x, linestyle='-', alpha=0.5, color='r', label='legend2')
    plt.draw()
    plt.pause(0.2)
    t2 = time.time()
    print("time_sapn2", t2 - t1)


def get_local_map(map_tp_arr, current_pos_tp):
    if len(map_tp_arr) > 0:
        map_data_arr_reshape = map_tp_arr.reshape((1000, 1000))
    else:
        map_data_arr_reshape = np.zeros((1000, 1000))

    current_pose_trans = convert_point_to_tpmap([current_pos_tp])
    # print('current_pose_trans', current_pose_trans)
    index_x = int(current_pose_trans[0][0])
    index_y = int(current_pose_trans[0][1])
    if (index_x >= 13) and (index_y >= 13) and (index_y < 987) and (index_x < 987):
        map_local = map_data_arr_reshape[(index_y-13):(index_y+14), (index_x-13):(index_x+14)]
    else:
        print('agent is corner')
        map_local = []
    return map_local




