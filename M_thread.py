import sys

from gt_tp_trans import *

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from action_ctrl import *
from mapper import *
from ros_thread import *
from path_plan import *
from orb_reset import *
import numpy.matlib

reset_flag = 0

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.distributions import MultivariateNormal
from network.network_TD3 import *
from typing import DefaultDict, List, Dict
from collections import defaultdict
import cv2
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import torch.optim as optim
from torch.distributions import Normal, Categorical

from torch.utils.tensorboard import SummaryWriter

import math

import openpyxl

import gc

# from network.network_SAC import Critic
from network.network_TD3 import *

import copy

import random


from std_msgs.msg import Bool
pubr_insert_key = rospy.Publisher('insert_key', Bool, queue_size=1)

is_cuda = 1
device0 = torch.device("cuda:0")
# device1 = torch.device("cpu")
pub_global_path = rospy.Publisher('glb_path', Path, queue_size=1)

def _to_tensor(v):
     if torch.is_tensor(v):
          return v
     elif isinstance(v, np.ndarray):
          return torch.from_numpy(v)
     else:
          return torch.tensor(v, dtype=torch.float)
          # return torch.as_tensor(v, dtype=torch.float)


#add the option!
class ReplayBuffer(object):
     def __init__(self):
          self.max_size = int(150)  
          self.count = 0
          self.size = 0

          self.depth_tensor = [3 for x in range(0,self.max_size)]
          self.angle_final_gt = [3 for x in range(0,self.max_size)]
          self.dist_abs_now_gt_float_tensor = [3 for x in range(0,self.max_size)]

          self.depth_tensor_next = [3 for x in range(0,self.max_size)]
          self.angle_final_gt_next = [3 for x in range(0,self.max_size)]
          self.dist_abs_now_gt_next_float_tensor = [3 for x in range(0,self.max_size)]

          
          # self.a = np.zeros((self.max_size, 2))
          self.a = [3 for x in range(0,self.max_size)]
          # self.r = np.zeros((self.max_size, 1))
          self.r = [3 for x in range(0,self.max_size)]
          self.p = [3 for x in range(0,self.max_size)]
          
          # self.s_a = [3 for x in range(0,self.max_size)]
          # self.out_dec = [3 for x in range(0,self.max_size)]
          
          self.option = [3 for x in range(0,self.max_size)]
          self.dw = np.zeros((self.max_size, 1))


     def store(self, p, option, depth_tensor, angle_final_gt, dist_abs_now_gt_float_tensor, depth_tensor_next, angle_final_gt_next, dist_abs_now_gt_next_float_tensor, action_, reward, dw):
          self.depth_tensor[self.count] = _to_tensor(depth_tensor.squeeze(0)).float()
          self.angle_final_gt[self.count] = _to_tensor(angle_final_gt.unsqueeze(0)).float()
          self.dist_abs_now_gt_float_tensor[self.count] = _to_tensor(dist_abs_now_gt_float_tensor.unsqueeze(0)).float()

          self.depth_tensor_next[self.count] = _to_tensor(depth_tensor_next.squeeze(0)).float()
          self.angle_final_gt_next[self.count] = _to_tensor(angle_final_gt_next.unsqueeze(0)).float()
          self.dist_abs_now_gt_next_float_tensor[self.count] = _to_tensor(dist_abs_now_gt_next_float_tensor.unsqueeze(0)).float()

          
          self.p[self.count] = _to_tensor(p)
          # self.s_a[self.count] = _to_tensor(s_a)
          # self.out_dec[self.count] = _to_tensor(out_dec)
          self.option[self.count] = option
          self.a[self.count] = action_
          self.r[self.count] = reward
          self.dw[self.count] = dw

          self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
          self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

     def sample(self, batch_size):
          index = np.random.choice(self.size, size=batch_size)  # Randomly sampling

          batch_depth_tensor = []
          for i in range(len(index)):
               batch_depth_tensor.append(self.depth_tensor[index[i]])
          batch_depth_tensor_t = torch.stack(batch_depth_tensor, dim=0).to(device=device0)

          batch_angle_final_gt = []
          for i in range(len(index)):
               batch_angle_final_gt.append(self.angle_final_gt[index[i]])
          batch_angle_final_gt_t = torch.stack(batch_angle_final_gt, dim=0).to(device=device0)

          batch_dist_abs_now_gt_float_tensor = []
          for i in range(len(index)):
               batch_dist_abs_now_gt_float_tensor.append(self.dist_abs_now_gt_float_tensor[index[i]])
          batch_dist_abs_now_gt_float_tensor_t = torch.stack(batch_dist_abs_now_gt_float_tensor, dim=0).to(device=device0)

          #===next===

          batch_depth_tensor_next = []
          for i in range(len(index)):
               batch_depth_tensor_next.append(self.depth_tensor_next[index[i]])
          batch_depth_tensor_next_t = torch.stack(batch_depth_tensor_next, dim=0).to(device=device0)

          batch_angle_final_gt_next = []
          for i in range(len(index)):
               batch_angle_final_gt_next.append(self.angle_final_gt_next[index[i]])
          batch_angle_final_gt_next_t = torch.stack(batch_angle_final_gt_next, dim=0).to(device=device0)

          batch_dist_abs_now_gt_next_float_tensor = []
          for i in range(len(index)):
               batch_dist_abs_now_gt_next_float_tensor.append(self.dist_abs_now_gt_next_float_tensor[index[i]])
          batch_dist_abs_now_gt_next_float_tensor_t = torch.stack(batch_dist_abs_now_gt_next_float_tensor, dim=0).to(device=device0)

          # batch_a = torch.tensor(self.a[index], dtype=torch.float).to(device=device0)
          
          batch_option = []
          for i in range(len(index)):
               batch_option.append(self.option[index[i]])
          batch_option = torch.stack(batch_option, dim=0).to(device=device0)

          batch_a = []
          for i in range(len(index)):
               batch_a.append(self.a[index[i]])
          batch_a_t = torch.stack(batch_a, dim=0).to(device=device0)

          # batch_r = torch.tensor(self.r[index], dtype=torch.float)
          batch_r = []
          for i in range(len(index)):
               batch_r.append(self.r[index[i]])
          batch_r_t = torch.stack(batch_r, dim=0).to(device=device0)
          
          batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(device=device0)
          
          batch_p = []
          for i in range(len(index)):
               batch_p.append(self.p[index[i]])
          batch_p = torch.stack(batch_p, dim=0).to(device=device0)
          
          
          # batch_s_a = []
          # for i in range(len(index)):
          #      batch_s_a.append(self.s_a[index[i]])
          # batch_s_a = torch.stack(batch_s_a, dim=0).to(device=device0)
          
          
          # batch_out_dec = []
          # for i in range(len(index)):
          #      batch_out_dec.append(self.out_dec[index[i]])
          # batch_out_dec = torch.stack(batch_out_dec, dim=0).to(device=device0)


          return  batch_p, batch_option, batch_depth_tensor_t, batch_angle_final_gt_t, batch_dist_abs_now_gt_float_tensor_t, batch_depth_tensor_next_t, batch_angle_final_gt_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_a_t, batch_r_t, batch_dw
     def clear(self):
          self.count = 0
          self.size = 0



class TD3:
     def __init__(self, policy_class = HRLTD3(2), **hyperparameters):

          self._init_hyperparameters(hyperparameters)

          self.act_dim = 2
          self.option_dim = 3
          
          self.actors = nn.ModuleList(Actor(self.act_dim).to(device0) for actor in range(self.option_dim)).to(device0)
          self.actors_target = copy.deepcopy(self.actors)

          self.critic = Critic().to(device0)
          self.critic_target = copy.deepcopy(self.critic)
          self.entropy_coeff = 0.1
          self.c_ent = 1
          self.option_update_size = 5
          
          self.option_dim = 3
          self.option_net = Option(self.option_dim).to(device0)

          self.actors_optimizer = torch.optim.Adam(self.actors.parameters(), lr=3e-4)
          self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
          self.option_optimizer = torch.optim.Adam(self.option_net.parameters(), lr=3e-4)


          if is_cuda:
               # self.actor.cuda()
               self.actors.to(device0)
               self.actors_target.to(device0)
               self.critic.to(device0)
               self.critic_target.to(device0)
               self.option_net.to(device0)
               
          self.rewad_arg1 = 100.0 #50.0 50.0 60
          self.rewad_arg2 = 1.0
          self.distance_goal = 0.6
          self.distance_local_goal = 0.23

          # self.model_path = "/home/jin/RL-code/orbslam_sim_old/perfect/option/TD3/model/date4_11_2/" + str(glovar.test) + "/"

          self.model_path = "/media/vision/data/wang/codes/TD3_HRL/model/date4_23_10/" + str(glovar.test) + "/"

          
          

          self.replay_buffer = ReplayBuffer()
          self.replay_buffer_on_policy = ReplayBuffer()

          # self.update_cycle = 4  #应该不需要，本来就是off-policy,batch_size调整大点就行
          
          self.batch_size = 64
          # self.batch_size = 4
          # self.batch_size = 1
          self.random_steps = 80  #replay_buffer能放200个吗？
          # self.random_steps = 2  #replay_buffer能放200个吗？

          # #===测试用===
          # self.batch_size = 5
          # self.random_steps = 5
          # #===测试用===

          self.GAMMA = 0.99  # discount factor
          self.TAU = 0.005
          self.update_target_interval = 5

          self.max_action = [0.5, 0.3]
          self.policy_noise = 0.2 * torch.tensor([0.5, 0.3]).to(device0)
          # self.noise_clip = 0.5 * self.max_action
          self.noise_clip_0 = 0.5 * self.max_action[0]
          self.noise_clip_1 = 0.5 * self.max_action[1]

          # self.actor_pointer = 0
          self.policy_freq = 2  # The frequency of policy updates
          self.repeat_times = 6
          
          
     # def softmax_option_target(self, depth, local_goal_point, angle, a):
          # Q_predict = []
          
          # for o in range(self.option_dim):
          #      action_i = self.actors_target(o)
          #      Q_predict_i, _ = self.critic_target(o, action_i)
          

          

     def learn(self, M_th, total_steps, writer):
          time.sleep(5.0)
          steps_so_far = 0  #
          init_num = 0

          while steps_so_far < total_steps:  # ALG STEP 2
               # torch.set_grad_enabled(False)
               goal_pos_gt = glovar.global_goal_gt
               cur_pos_gt = M_th.agent.state.position
               threadLock.acquire()
               _, cur_pos_tp2 = get_current_pose(glovar.camera_trajectory)  #加线程锁
               cur_pos_tp = deepcopy(cur_pos_tp2)
               threadLock.release()
               # _, cur_pos_tp = get_current_pose(glovar.camera_trajectory)
               for i in range(4):
                    if len(cur_pos_tp) == 0:
                         time.sleep(2.0)
                         threadLock.acquire()
                         _, cur_pos_tp2 = get_current_pose(glovar.camera_trajectory)  #加线程锁
                         cur_pos_tp = deepcopy(cur_pos_tp2)
                         threadLock.release()
                         # _, cur_pos_tp = get_current_pose(glovar.camera_trajectory)
                         if i == 2:
                              print('cur_pos_tp not received')
                              print('start_re_init')
                              while init_num < 360:
                                   action(M_th.agent, 2)
                                   time.sleep(0.5)

                                   threadLock.acquire()
                                   _, cur_pos_tp2 = get_current_pose(glovar.camera_trajectory)  #加线程锁
                                   cur_pos_tp = deepcopy(cur_pos_tp2)
                                   threadLock.release()
                                   # _, cur_pos_tp = get_current_pose(glovar.camera_trajectory)
                                   init_num += 1
                                   if len(cur_pos_tp) != 0:
                                        # actiopn_n(M_th.agent, 2, 20)
                                        # collided = actiopn_n_vel_start(M_th.agent, 2, M_th.vel_control, M_th.time_step, M_th.sim_step_filter, num=50)  # 连续动作
                                        break
                    else:
                         q = M_th.agent.state.rotation
                         t_ini = M_th.agent.state.position
                         # print('t_ini', t_ini)
                         T = q_T(q, t_ini)
                         M_th.R_ini = T.rotation
                         M_th.t_ini = T.position
                         break

               goal_pos_tp = get_goalpoint_tp(M_th.R_ini, M_th.t_ini, cur_pos_gt, goal_pos_gt, cur_pos_tp)
               self.rollout(M_th, goal_pos_gt, goal_pos_tp, writer)  # ALG STEP 3

               glovar.sim_reset = 1
               time.sleep(0.5)
               print ('glovar.thread_1: ', len(glovar.thread))
               if glovar.change_flag == 1:
                    # print('time.sleep(10)')
                    # time.sleep(5)   
                    # print('end time.sleep(10)')
                    break
               else:
                    time.sleep(2.0)
                    q = M_th.agent.state.rotation
                    t_ini = M_th.agent.state.position
                    # print('t_ini', t_ini)
                    T = q_T(q, t_ini)
                    M_th.R_ini = T.rotation
                    M_th.t_ini = T.position

     def rollout(self, M_th, goal_pos_gt, goal_pos_tp, writer):

          collided = 0
          track_flag = 0
          glovar.train_global_num += 1
          glovar.local_steps = 0

          while True:
               print ('glovar.thread_2: ', len(glovar.thread))
               if collided or track_flag:

                    # gc.collect()
                    break

               cur_pos_gt = deepcopy(M_th.agent.state.position)
               dis_goal = distance(cur_pos_gt, goal_pos_gt)
               print('The distance between cur_pos_gt and goal_pos_gt is ', dis_goal)
               glovar.train_num += 1
               if dis_goal > self.distance_goal:
                    glovar.train_local_num += 1

                    #规划局部目标点
                    threadLock.acquire()
                    pose = glovar.camera_trajectory
                    _, cur_pos_tp2 = get_current_pose(pose)  #加线程锁
                    cur_pos_tp = deepcopy(cur_pos_tp2)
                    current_pos_tp_map2 = get_current_pose_tp(pose)  #tp图上当前点的位置
                    current_pos_tp_map = deepcopy(current_pos_tp_map2)
                    threadLock.release()

                    goal_pos_tp = get_goalpoint_tp(M_th.R_ini, M_th.t_ini, cur_pos_gt, goal_pos_gt, cur_pos_tp)
                    goal_pos_map = convert_point_to_tpmap(goal_pos_tp.reshape([1, 3]), 0.04)
                    goal_pose_tensor = torch.as_tensor(goal_pos_map)  #tp图上目标点的位置
                    # print ('goal_pose_tensor: ', goal_pose_tensor)
                    current_pos_tp_map[0][1] = current_pos_tp_map2[0][0]
                    current_pos_tp_map[0][0] = current_pos_tp_map2[0][1]

                    image_tensor, depth_tensor, mpt_tensor, angle_tensor_tp, obv_sts, map_data, mpt_num = M_th.get_states()

                    
                    while (len(np.array(map_data)) == 0):
                         print('map_data is None, recevive again')
                         # time.sleep(1.0)
                         image_tensor, depth_tensor, mpt_tensor, angle_tensor_tp, obv_sts, map_data, mpt_num = M_th.get_states()                    
                    
                    path, cost, plan_way_points = plan_path(np.array(map_data), current_pos_tp_map,
                                                        goal_pose_tensor,
                                                        height=1000, width=1000)
                    
                    glb_path = Path()
                    glb_path.header.frame_id = 'world'
                    glb_path.header.stamp = rospy.Time.now()
                    # path = []
                    for item in path:
                         glb_this_pos = PoseStamped()
                         glb_this_pos.header.frame_id = 'world'
                         glb_this_pos.header.stamp = rospy.Time.now()
                         glb_this_pos.pose.position.x = item[0] * 0.04 - 20
                         glb_this_pos.pose.position.y = item[1] * 0.04 - 20
                         glb_this_pos.pose.orientation.x = 0
                         glb_this_pos.pose.orientation.y = 0
                         glb_this_pos.pose.orientation.z = 0
                         glb_this_pos.pose.orientation.w = 0
                         glb_path.poses.append(glb_this_pos)
                         # path += [torch.tensor(item).to(device)]
                    pub_global_path.publish(glb_path)
                    local_goal_ = get_local_goal_point(path, current_pos_tp_map)
                    # print ('local_goal_: ', local_goal_)

                    #以local_goal_为局部目标
                    #TP坐标系下计算相对距离
                    # print ('type(local_goal_): ', type(local_goal_))
                    # print ('type(cur_pos_tp): ', type(cur_pos_tp))
                    local_goal_tp = deepcopy(cur_pos_tp)
                    local_goal_tp[0] = local_goal_[0]*0.04-20
                    local_goal_tp[2] = local_goal_[1]*0.04-20
                    local_goal_tp_array = np.array(local_goal_tp)
                    dist_ = local_goal_tp_array - cur_pos_tp
                    dist_ = dist_.astype(float)
                    dist_abs_now_tp = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
                    # batch_dist_abs_now_tp_log.append(dist_abs_now_tp)  #tp坐标系下，当前位置和局部点之间的距离
                    #TP坐标系下在地图上计算相对角度
                    current_pos_tp_ = [current_pos_tp_map[0][0], current_pos_tp_map[0][1]]
                    angle_final = angle_TP(local_goal_, current_pos_tp_, angle_tensor_tp)
                    angle_final_tp = angle_final
                    angle_tp_now = angle_tensor_tp.detach().cpu().numpy()
                    angle_tp_now = float(angle_tp_now)

                    #plot gt map
                    current_pos_gt = deepcopy(M_th.agent.state.position)
                    # print ('current_pos_gt: ', current_pos_gt)
                    current_pos_gt2 = deepcopy(current_pos_gt)
                    local_goal_gt = map2gt(M_th.R_ini, M_th.t_ini, local_goal_, current_pos_gt2, cur_pos_tp)
                    # vis_points = [M_th.agent.state.position]
                    # vis_points.append(local_goal_gt)
                    # vis_points.append(goal_pos_gt)
                    # xy_vis_points = convert_points_to_topdown(M_th.sim_pathfinder, vis_points)
                    # height = M_th.agent.state.position[1]
                    # sim_topdown_map = M_th.sim_pathfinder.get_topdown_view(0.04, height)
                    # display_map2(sim_topdown_map, 0, key_points=xy_vis_points)

                    #GT坐标系下在地图上计算和局部目标点之间的相对距离
                    current_pos_gt = deepcopy(M_th.agent.state.position)
                    # print ('current_pos_gt: ', current_pos_gt)
                    dist_ = local_goal_gt - current_pos_gt
                    dist_ = dist_.astype(float)
                    dist_abs_now_gt = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
                    # print ('dist_gt: ', dist_abs_now_gt)

                    #GT坐标系下计算和局部目标点之间的相对角度
                    cur_oritation_gt = M_th.agent.state.rotation
                    cur_oritation_gt_ = np.array([cur_oritation_gt.w, cur_oritation_gt.x, cur_oritation_gt.y, cur_oritation_gt.z])
                    angle_gt_now = get_angle_gt(cur_oritation_gt_)
                    angle_tensor_tensor = torch.tensor(angle_gt_now, dtype=torch.float)
                    angle_tensor_gt = angle_tensor_tensor.to(device0)
                    angle_final_gt = angle_GT(local_goal_gt, current_pos_gt, angle_tensor_gt)  #角度

                    t = 0

               
               else:
                    print('gobal_goal_point has arrived')
                    glovar.global_success_num += 1
                    glovar.success = 1


                    # gc.collect()
                    break
               
               option_done = False     
               while t < self.steps_per_local_goal:
                    print ('start!!!!start!!!!start!!!!start!!!!start!!!!start!!!!')

                    glovar.random += 1
                    print ('glovar.random: ', glovar.random)

                    t += 1
                    glovar.local_steps += 1

                    # print ('image_tensor.shape: ', image_tensor.shape)   #torch.Size([1, 3, 320, 240])

                    #奖励用的是GT，优化时用的数据也必须是GT
                    # batch_angle_gt_now_log.append(angle_gt_now)  ##记录gt坐标系下，机器人当前的朝向
                    dist_abs_now_gt_float = dist_abs_now_gt.astype(float)
                    dist_abs_now_gt_float_tensor = torch.tensor(dist_abs_now_gt_float)
                    writer.add_scalar('dist/Step', dist_abs_now_gt_float_tensor.item(), glovar.random)

                    writer.add_scalar('angle/Step', angle_final_gt.item(), glovar.random)

                    
                    # #===动作===
                    # action_ = self.get_action(depth_tensor, dist_abs_now_gt_float_tensor,
                    #                                 angle_final_gt)
                    # action_step = np.random.rand(1, 2)
                    # action_step[0,0] = action_[0,0] - 0.25
                    # action_step[0,1] = action_[0,1]
                    # print ('action_step: ', action_step)
                    # # print ('action_: ', action_)  #1
                    # # # # print ('action_.shape: ', action_.shape)

                    if option_done == False:
                         option, _, Q_predict = self.softmax_option_target(depth_tensor, dist_abs_now_gt_float_tensor, angle_final_gt)
                              
                         option = option[0, 0]
                         option_done = True
                    if glovar.random < self.random_steps:
        
                         print ('random action!')
                         action_ = np.random.rand(2)
                         # action_step = np.random.rand(1, 2)
                         # number1 = random.random()
                         number1 = 2 * random.random() - 1
                         action_[0] = 0.5 * number1
                         # action_step[0,0] = (-0.5) * number1
                         number2 = 2 * random.random() - 1
                         action_[1] = 0.3 * number2
                         p = 1
                         # action_step[0,1] = 0.3 * number2
                         # action_ = action_.astype(float)
                    else:
                         # action_ = self.get_action(depth_tensor, dist_abs_now_gt_float_tensor,
                         #                            angle_final_gt)
                         # if(t % 5 == 0):
                         if(t % 6 == 0):
                              option, _, Q_predict = self.softmax_option_target(depth_tensor, dist_abs_now_gt_float_tensor, angle_final_gt)
                              
                              option = option[0, 0]
                         
                         # print('option: ', option)
                         action_ = self.get_action(depth_tensor, dist_abs_now_gt_float_tensor, angle_final_gt, option)
                         
                         
                         
                         # print('Q_predict.shape: ', Q_predict.shape)
                         p = torch.softmax(Q_predict, dim=1)[0][int(option)]
                         p = p.cpu().detach().numpy()
                         # print('p: ', p)
                         # p = np.asarray(torch.softmax(Q_predict.cpu().detach().numpy())[0][option])
                         # right or not can't understand
                         
                         # noise = torch.randn_like(action_)??????????????????????????????????????????
                         
                         # action_[0] = (action_[0] + 0.1 * 0.5).clip(-0.5, 0.5)
                         action_[0] = (action_[0] + np.random.normal(0, 0.1 * 0.5)).clip(-0.5, 0.5)
                         action_[1] = (action_[1] + np.random.normal(0, 0.1 * 0.3)).clip(-0.3, 0.3)

                    #      # action_[0,0] = action_[0,0] - 0.25
                    #      action_step = np.random.rand(1, 2)
                    #      action_step[0,0] = action_[0,0] - 0.25
                    #      action_step[0,1] = action_[0,1]
                    # print ('action_step: ', action_step)
                    print ('action_: ', action_)   #[0.46324125 0.07810053]


                    image_tensor_next, depth_tensor_next, mpt_tensor_next, angle_final_gt_next, reward, done, collided, reward_dist_log_gt, dist_abs_now_gt_next, angle_gt_now, dist_abs_now_tp, angle_final_tp, angle_tp_now, mpt_num, reward_mpt_log = self.step(action_, M_th,
                                                                    goal_pos_gt, angle_final_gt, dist_abs_now_gt, local_goal_gt, t, local_goal_, mpt_num)

                    threadLock.acquire()
                    cur_track_sts = glovar.orb_sts_mpt['sts']
                    threadLock.release()
                    if cur_track_sts is not 2:
                         print ('traking failure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                         track_flag = 1
                         # reward = -20.0  #-60, -30
                         reward = -10.0  #没有角度奖励
                         # reward = torch.tensor(reward).unsqueeze(0).float().to(device0)

                         #logging
                         glovar.global_tr_num += 1
                         break
                    
                    if collided:
                         print ('collided happen!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                         # reward = -20.0  #60. 30
                         reward = -10.0  #没有角度奖励
                         # reward = torch.tensor(reward).unsqueeze(0).float().to(device0)
                         glovar.global_collision_num += 1
                         break


                    dw = False
                    if done:  # 局部目标点完成
                         print('local_goal_point has arrived')
                         glovar.local_success_num += 1
                         dw = True
                         break
                    
                    #logging
                    glovar.total_t += 1
                    # print ('total_t: ', glovar.total_t)

                    if not os.path.exists(self.model_path):
                         os.makedirs(self.model_path)

                    #===测试用注释===
                    # Save our model if it's time
                    if glovar.total_t % 100 == 0:
                         print ('=======save model!!!=========')
                         torch.save(self.actors.state_dict(), self.model_path + 'TD3_actor.pth')
                         torch.save(self.actors_target.state_dict(), self.model_path + 'TD3_actor_target.pth')
                         torch.save(self.critic.state_dict(), self.model_path + 'TD3_critic.pth')
                         torch.save(self.critic_target.state_dict(), self.model_path + 'TD3_critic_target.pth')
                    #===测试用注释===

                    reward = torch.tensor(reward).unsqueeze(0).float().to(device0)
                    # print ('reward: ', reward)
                    writer.add_scalar('reward/Step', reward.item(), glovar.random)
                    
                    dist_abs_now_gt_next_float = dist_abs_now_gt_next.astype(float)
                    dist_abs_now_gt_next_float_tensor = torch.tensor(dist_abs_now_gt_next_float)
                    
                    action_ = torch.tensor(action_).squeeze(0).float().to(device0)
                    # print ('action_.shape: ', action_.shape)
                    #===测试时要用注释===
                    print ('option: ', option)
                    option = torch.tensor(option).squeeze(0).float().to(device0)

                    # print('option.shape: ', option.shape)
                    self.replay_buffer.store(p, option, depth_tensor, angle_final_gt, dist_abs_now_gt_float_tensor, depth_tensor_next, angle_final_gt_next, dist_abs_now_gt_next_float_tensor, action_, reward, dw)
                    self.replay_buffer_on_policy.store(p, option, depth_tensor, angle_final_gt, dist_abs_now_gt_float_tensor, depth_tensor_next, angle_final_gt_next, dist_abs_now_gt_next_float_tensor, action_, reward, dw)
                    #===测试时要用注释===

                    # image_tensor = image_tensor_next
                    depth_tensor = depth_tensor_next
                    # mpt_tensor = mpt_tensor_next
                    angle_final_gt = angle_final_gt_next
                    dist_abs_now_gt = dist_abs_now_gt_next


                    # if (self.replay_buffer.size > 3):
                    #      batch_depth_tensor_t, batch_angle_final_gt_t, batch_dist_abs_now_gt_float_tensor_t, batch_depth_tensor_next_t, batch_angle_final_gt_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_a, batch_r_t, batch_dw = self.replay_buffer.sample(3)
                    #      print ('test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #      print ('batch_a: ', batch_a)
                    #      print ('batch_a.shape: ', batch_a.shape)  #torch.Size([3, 1])

                    #===应该是单步更新的off-policy
                    if (self.replay_buffer.size > self.batch_size):

                         for u in range(int(self.repeat_times)):
                              # self.actor_pointer += 1
                              
                              if(self.replay_buffer_on_policy.size > self.option_update_size):
                                   print ('evaluate option!!!!!!!!!!!!')
                                   batch_p, batch_option, batch_depth_tensor_t, batch_angle_final_gt_t, batch_dist_abs_now_gt_float_tensor_t, batch_depth_tensor_next_t, batch_angle_final_gt_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_a, batch_r_t, batch_dw = self.replay_buffer_on_policy.sample(self.batch_size)
                                   
                                   
                                   with torch.no_grad():
                                        next_option_batch, _, Q_predict_next = self.softmax_option_target(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t)
                                        _, batch_s_a, batch_out_dec = self.option_net(batch_depth_tensor_t, batch_dist_abs_now_gt_float_tensor_t, batch_angle_final_gt_t, batch_a)
                                        # noise_clip = 0.5
                                        noise = torch.randn_like(batch_a)
                                        noise[:,0] = (torch.randn_like(batch_a) * self.policy_noise)[:,0].clamp(-self.noise_clip_0, self.noise_clip_0)
                                        noise[:,1] = (torch.randn_like(batch_a) * self.policy_noise)[:,1].clamp(-self.noise_clip_1, self.noise_clip_1)
                                        # print ('noise.shape: ', noise.shape)
                                        # print('next_option_batch', next_option_batch.shape)
                                        # print('batch_depth_tensor_next_t', batch_depth_tensor_next_t.shape)
                                        next_option_batch = torch.tensor(next_option_batch).squeeze(0).float().to(device0)
                                        # print('next_option_batch', next_option_batch.shape)
                                        # print('noise', noise.shape)

                                        next_action = self.predict_actor_target(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t, next_option_batch) 
                                        # print('next_action.shape', next_action.shape)
                                        next_action = next_action + noise

                                        target_Q1, target_Q2 = self.critic_target(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t, next_action)
                                        target_Q = batch_r_t + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

                                        batch_option, _, _1 = self.option_net(batch_depth_tensor_t, batch_dist_abs_now_gt_float_tensor_t, batch_angle_final_gt_t, batch_a)
                                        predicted_v_i = self.value_function(batch_depth_tensor_t, batch_dist_abs_now_gt_float_tensor_t, batch_angle_final_gt_t)
                                   
                                        Advantage = target_Q - predicted_v_i
                                        # print ('Advantage: ', Advantage.shape)
                                        batch_p = batch_p.view(-1, 1)
                                        # print('batch_p', batch_p)
                                        weight = torch.divide(torch.exp(Advantage - torch.max(Advantage)), batch_p)
                                        # print ('weight: ', weight.shape)
                                        w_norm = weight / torch.mean(weight)
                                        # print ('w_norm: ', w_norm.shape)
                                        # print ('batch_option: ', batch_option.shape)
                                        
                                        entropy = torch.sum(w_norm * batch_option * torch.log(batch_option.add(1e-8)))
                                        P_weightd_ave = torch.mean(torch.multiply(w_norm,batch_option))
                                        
                                        critic_entropy = entropy - self.c_ent * torch.sum(P_weightd_ave * torch.log(P_weightd_ave.add(1e-8)))
                                   reg_loss = F.mse_loss(batch_s_a, batch_out_dec)
                                   option_loss = reg_loss + self.entropy_coeff * critic_entropy
                                   option_loss.requires_grad = True
                                   print ('option_loss: ', option_loss)
                                   self.option_optimizer.zero_grad()
                                   option_loss.backward()
                                   self.option_optimizer.step()
                                   # self.replay_buffer_on_policy.clear()
                              
                              
                              
                              
                              print ('evaluate!!!!!!!!!!!!')
                              
                              
                              
                              
                              
                              
                              batch_p, batch_option, batch_depth_tensor_t, batch_angle_final_gt_t, batch_dist_abs_now_gt_float_tensor_t, batch_depth_tensor_next_t, batch_angle_final_gt_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_a, batch_r_t, batch_dw = self.replay_buffer.sample(self.batch_size)
                         #      # print ('batch_depth_tensor_t.shape: ', batch_depth_tensor_t.shape)
                         #      # print ('batch_angle_final_gt_t.shape: ', batch_angle_final_gt_t.shape)
                         #      # print ('batch_dist_abs_now_gt_float_tensor_t.shape: ', batch_dist_abs_now_gt_float_tensor_t.shape)
                         #      # print ('batch_depth_tensor_next_t.shape: ', batch_depth_tensor_next_t.shape)
                         #      # print ('batch_angle_final_gt_next_t.shape: ', batch_angle_final_gt_next_t.shape)
                         #      # print ('batch_dist_abs_now_gt_next_float_tensor_t.shape: ', batch_dist_abs_now_gt_next_float_tensor_t.shape)
                         #      # print ('batch_a.shape: ', batch_a.shape)   #torch.Size([5, 2])
                         #      # print ('batch_r_t.shape: ', batch_r_t.shape)
                         #      # print ('batch_dw.shape: ', batch_dw.shape)

                              # Compute the target Q
                              with torch.no_grad():  # target_Q has no gradient
                                   # Trick 1:target policy smoothing
                                   # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
                                   noise = torch.randn_like(batch_a)
                                   noise[:,0] = (torch.randn_like(batch_a) * self.policy_noise)[:,0].clamp(-self.noise_clip_0, self.noise_clip_0)
                                   noise[:,1] = (torch.randn_like(batch_a) * self.policy_noise)[:,1].clamp(-self.noise_clip_1, self.noise_clip_1)
                                   # print ('noise.shape: ', noise.shape)
                                   next_option_batch, _, __ = self.softmax_option_target(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t)
                                   next_action = self.predict_actor_target(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t, next_option_batch) + noise
                                   # print ('next_action.shape: ', next_action.shape)
                                   next_action[:,0] = next_action[:,0].clamp(-self.max_action[0], self.max_action[0])
                                   next_action[:,1] = next_action[:,1].clamp(-self.max_action[1], self.max_action[1])
                                   # print ('next_action.shape: ', next_action.shape)

                                   # Trick 2:clipped double Q-learning
                                   target_Q1, target_Q2 = self.critic_target(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t, next_action)
                                   target_Q = batch_r_t + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)


                              # Get the current Q
                              current_Q1, current_Q2 = self.critic(batch_depth_tensor_t, batch_dist_abs_now_gt_float_tensor_t, batch_angle_final_gt_t, batch_a)
                              # Compute the critic loss
                              critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                              # Optimize the critic
                              print("critic_loss",critic_loss)
                              self.critic_optimizer.zero_grad()
                              critic_loss.backward()
                              self.critic_optimizer.step()
                              writer.add_scalar('critic_loss/Step', critic_loss.item(), glovar.random)

                              # Trick 3:delayed policy updates
                              # if self.actor_pointer % self.policy_freq == 0:
                              
                              
                              ###########################################################
                              if u % self.policy_freq == 0:
                                   # Freeze critic networks so you don't waste computational effort
                                   for params in self.critic.parameters():
                                        params.requires_grad = False
                                        
                                   option_estimated, _1, _2 = self.option_net(batch_depth_tensor_t, batch_dist_abs_now_gt_float_tensor_t, batch_angle_final_gt_t, batch_a)
                                   
                                   max_index = torch.argmax(option_estimated, dim=1).int()
                                   # print("max_index",max_index)
                                   for o, actor in zip(range(self.option_dim), self.actors):
                                        
                                        indx_o = (max_index == o)   #用于筛选！
                                        is_null = torch.all(indx_o == False)
                                        # print("indx_o",indx_o)
                                        # print("is_null",is_null)
                                        if is_null:
                                             continue
                                        
                                        
                                        
                                        # Compute actor loss
                                        # print("batch_depth_tensor[indx_o]",batch_depth_tensor_t[indx_o].shape)
                                        action_up = actor(batch_depth_tensor_t[indx_o], batch_dist_abs_now_gt_float_tensor_t[indx_o], batch_angle_final_gt_t[indx_o])
                                        actor_loss = -self.critic.Q1(batch_depth_tensor_t[indx_o], batch_dist_abs_now_gt_float_tensor_t[indx_o], batch_angle_final_gt_t[indx_o], action_up).mean()  # Only use Q1
                                        # Optimize the actor
                                        print("actor_loss",actor_loss)
                                        self.actors_optimizer.zero_grad()
                                        
                                        actor_loss.backward()
                                        self.actors_optimizer.step()
                                        writer.add_scalar('actor_loss/Step', actor_loss.item(), glovar.random)

                                   # Unfreeze critic networks
                                   for params in self.critic.parameters():
                                        params.requires_grad = True
                                   
                                   # Softly update the target networks
                                   for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                                        target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                                   for param, target_param in zip(self.actors.parameters(), self.actors_target.parameters()):
                                        target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
                                   
                         self.replay_buffer_on_policy.clear()



     def softmax_option_target(self, depth, local_goal_point, angle):
     
          Q_predict = []
          n = depth.shape[0]
          for o in range(self.option_dim):
               action_i = self.predict_actor_target(depth, local_goal_point, angle, o)
               # print('action_i.shape',action_i.shape)
               Q_predict_i, _ = self.predict_critic_target(depth, local_goal_point, angle, action_i)
               
               if o == 0 :
                    Q_predict = Q_predict_i.view(-1, 1)
               else:
                    Q_predict = torch.cat((Q_predict, Q_predict_i.view(-1, 1)), dim = 1)
                    
          p = torch.softmax(Q_predict, dim = 1)
          p = p.cpu().detach().numpy()
          row, col = p.shape


          p_sum = np.reshape(np.sum(p, axis=1), (row, 1))
          # print('p_sum.shape: ', p_sum.shape)
          p_normalized = p  / np.matlib.repmat(p_sum, 1, col)
          # print('p_normalized.shape: ', p_normalized.shape)
          p_cumsum = np.matrix(np.cumsum( p_normalized, axis=1))
          # print(p_cumsum[0])
          rand = np.matlib.repmat(np.random.random((row, 1)), 1, col)
          # print(rand[0])
          o_softmax = np.argmax(p_cumsum >= rand, axis=1)
     

          
          n = Q_predict.shape[0]
          
          Q_softmax = Q_predict[np.arange(n), o_softmax.flatten()]
          
          return o_softmax, Q_softmax.view(n, 1), Q_predict
               

               
               
     
     
     def predict_critic_target(self, depth, local_goal_point, angle, a):
          return self.critic_target(depth, local_goal_point, angle, a)
     
     def predict_critic(self, depth, local_goal_point, angle, a):
          return self.critic(depth, local_goal_point, angle, a)
          
     
     
     
     def predict_actor_target(self, depth, local_goal_point, angle, option):
          action_list = []
          for actor_target in self.actors_target:
               action_o = actor_target(depth, local_goal_point, angle)
               # print('action_o.shape',action_o.shape)
               action_list.append(action_o)
          
     
          # print('action_list.shape',len(action_list))
          n = depth.shape[0]
          # print('n',n)
          # print('option.shape',option.shape)
          # n is the batch size actually
          
          if n ==1 or np.isscalar(option):
               action =action_list[option]
               # print('action.shape',action.shape)
          else:
               for i in range(n):
                    if i == 0:
                         action = action_list[int(option[i])][i, :]
                    # print('action.shape',action.shape)
                    else:
                         action = torch.vstack((action, action_list[int(option[i])][i, :]))
                         # print('action.shape',action.shape)
                    # action = np.vstack(action, action_list[option[i]][i, :])
          return action
     
     def predict_actor(self, depth, local_goal_point, angle, option):
          action_list = []
          for actor in self.actors:
               action_o = actor(depth, local_goal_point, angle)
               action_list.append(action_o)
          
          action_list = torch.cat(action_list, 1)
          n = depth.shape[0]
          # n is the batch size actually
          
          if n ==1 or np.isscalar(option):
               action =action_list[option]
          else:
               for i in range(n):
                    if i == 0:
                         action = action_list[int(option[i])][i, :]
                    else:
                         action = torch.vstack(action, action_list[int(option[i])][i, :])
                    # action = np.vstack(action, action_list[option[i]][i, :])
          return action
     
     def value_function(self, depth, local_goal_point, angle):
          Q_predict = []
          n = depth.shape[0]
         
          for o, actor in zip(range(self.option_dim), self.actors_target):
               action_i = actor(depth, local_goal_point, angle)
               Q_predict1, Q_predict2 = self.critic_target(depth, local_goal_point, angle, action_i)
               Q_predict_i = torch.min(Q_predict1, Q_predict2)
               
               if o == 0:
                    Q_predict = Q_predict_i.view(-1, 1)
               else:
                    Q_predict = torch.cat((Q_predict, Q_predict_i.view(-1, 1)), dim = 1)
          
          po = torch.softmax(Q_predict, dim = 1)
          weigth_mean = torch.mean(po, dim = 1)
          x_weighted = torch.multiply(Q_predict, po)
          mean_weight = torch.divide(torch.mean(x_weighted, dim = 1), weigth_mean)
          
          state_values = mean_weight.view(-1, 1)
          
          return state_values
                   
          
     
     
     
     
     
     def soft_update(self, net, target_net):
          for param_target, param in zip(target_net.parameters(),
                                        net.parameters()):
               param_target.data.copy_(param_target.data * (1.0 - self.TAU) +
                                        param.data * self.TAU)

     def _update_target_network(self, current_net: nn.Module, target_net: nn.Module):
          for target_param, param in zip(target_net.parameters(), current_net.parameters()):
               target_param.data.copy_(
                    self.TAU * param.data + (1 - self.TAU) * target_param.data)
     
     def get_action(self, depth_tensor, local_goal_point, angle, option):
          
          actor = self.actors[int(option)]
          
          a = actor(depth_tensor, local_goal_point, angle).data.cpu().numpy().flatten()
          return a

     
     def calc_target(self, batch_r_t, batch_depth_tensor_next_t, batch_angle_final_gt_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_dw):
          next_actions, log_prob = self.actor(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t)
          entropy = -log_prob
          q1_value = self.target_critic1(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t, next_actions)
          q2_value = self.target_critic2(batch_depth_tensor_next_t, batch_dist_abs_now_gt_next_float_tensor_t, batch_angle_final_gt_next_t, next_actions)
          next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
          td_target = batch_r_t + self.GAMMA * next_value * (1 - batch_dw)


          return td_target


     
     def step(self, action_, M_th, goal_pos_gt, angle_final_last, dist_gt_last, local_goal_gt2, t, local_goal_, mpt_num):
          # if (action_ == 1):
          #      move = 'move_forward!!!'
          #      print (move)
          # elif (action_ == 0):
          #      move = 'turn_left!!!'
          #      print (move)
          # elif (action_ == 2):
          #      move = 'turn_right!!!'
          #      print (move)
          
          # # #正常的走
          # # max_action = 0
          # # max_action_num = 0
          # # for i in range(3):
          # #      if action_[0][0][i] > max_action:
          # #           max_action = action_[0][0][i]
          # #           max_action_num = i
          # # if (max_action_num == 1):
          # #      move = 'move_forward!!!'
          # #      print (move)
          # # elif (max_action_num == 0):
          # #      move = 'turn_left!!!'
          # #      print (move)
          # # elif (max_action_num == 2):
          # #      move = 'turn_right!!!'
          # #      print (move)
          
          # collided = actiopn_n_vel(M_th.agent, action_, M_th.vel_control, M_th.time_step, M_th.sim_step_filter,
          #                        num=3)  # 连续动作
          collided = actiopn_vel_new(M_th.agent, action_, M_th.vel_control, M_th.time_step, M_th.sim_step_filter)  # 连续动作
          pubr_insert_key.publish(1)

          #重新设置local_goal_gt
          #获取TP坐标系下的当前位置
          time.sleep(0.4)
          threadLock.acquire()
          _, cur_pos_tp2 = get_current_pose(glovar.camera_trajectory)  #加线程锁
          cur_pos_tp = deepcopy(cur_pos_tp2)
          threadLock.release()
          cur_pos_gt = deepcopy(M_th.agent.state.position)
          local_goal_gt = map2gt(M_th.R_ini, M_th.t_ini, local_goal_, cur_pos_gt, cur_pos_tp)

          #GT坐标系下在地图上计算和局部目标点之间的相对距离
          current_pos_gt = deepcopy(M_th.agent.state.position)


          #GT坐标系下计算和局部目标点之间的相对角度
          cur_oritation_gt = M_th.agent.state.rotation
          cur_oritation_gt_ = np.array([cur_oritation_gt.w, cur_oritation_gt.x, cur_oritation_gt.y, cur_oritation_gt.z])
          angle_gt_now = get_angle_gt(cur_oritation_gt_)
          angle_tensor_tensor = torch.tensor(angle_gt_now, dtype=torch.float)
          angle_tensor_gt = angle_tensor_tensor.to(device0)
          angle_final_gt = angle_GT(local_goal_gt, current_pos_gt, angle_tensor_gt)  #角度

          #如果角度大于30度或者小于-30度，那么前进奖励就小
          if ((angle_final_gt > 30) and (angle_final_last >= 30)) or ((angle_final_gt < -30) and (angle_final_last <= -30)):
          # if (angle_final_gt >= -30) and (angle_final_gt <= 30) and (angle_final_last >= -30) and (angle_final_last <= 30):
               print ('=======angle is large, Turn angle!!!Turn angle!!!Turn angle!!!')
               self.rewad_arg1 = 100
               dist_ = local_goal_gt - current_pos_gt
               dist_ = dist_.astype(float)
               dist_abs_now_gt = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
               # print ('dist_gt: ', dist_abs_now_gt)
               #计算GT坐标系下的相对距离的奖励
               reward_dist_gt = (dist_gt_last - dist_abs_now_gt) * self.rewad_arg1 - 0.01
               print ('dist_gt_last: ', dist_gt_last)
               print ('dist_abs_now_gt: ', dist_abs_now_gt)
               # print ('reward_dist_gt: ', reward_dist_gt)
               reward = reward_dist_gt
               reward_dist_log_gt = float(reward_dist_gt)

               #======角度奖励部分=======
               print ('angle_final_gt: ', angle_final_gt)
               print ('angle_final_last: ', angle_final_last)

               if (angle_final_gt > 0) and (angle_final_last >= 0):
                    reward_angle = (angle_final_last - angle_final_gt) * self.rewad_arg2
               elif (angle_final_gt > 0) and (angle_final_last < 0):
                    if (torch.abs(angle_final_gt) == torch.abs(angle_final_last)):
                         reward_angle = 0
                    elif (torch.abs(angle_final_gt) < torch.abs(angle_final_last)):
                         reward_angle = -(angle_final_last + angle_final_gt) * self.rewad_arg2
                    else:
                         reward_angle = -(angle_final_last + angle_final_gt) * self.rewad_arg2
               elif (angle_final_gt == 0):
                    reward_angle = torch.abs(angle_final_last) * self.rewad_arg2
               elif (angle_final_gt < 0) and (angle_final_last >= 0):
                    if (torch.abs(angle_final_gt) == torch.abs(angle_final_last)):
                         reward_angle = 0
                    else:
                         reward_angle = (angle_final_last + angle_final_gt) * self.rewad_arg2
               elif (angle_final_gt < 0) and (angle_final_last < 0):
                    reward_angle = (angle_final_gt - angle_final_last) * self.rewad_arg2

               print ('reward_angle: ', reward_angle)
               reward += reward_angle

               # reward_angle_log = reward_angle.detach().cpu().numpy()
               # reward_angle_log = float(reward_angle_log)
               #======角度奖励部分=======
          #如果角度在-30度和30度之间，就没有角度奖励
          else:
               print ('=======angle is small, Move!!!Move!!!Move!!!')
               self.rewad_arg1 = 100
               dist_ = local_goal_gt - current_pos_gt
               dist_ = dist_.astype(float)
               dist_abs_now_gt = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
               # print ('dist_gt: ', dist_abs_now_gt)
               #计算GT坐标系下的相对距离的奖励
               reward_dist_gt = (dist_gt_last - dist_abs_now_gt) * self.rewad_arg1 - 0.01
               print ('dist_gt_last: ', dist_gt_last)
               print ('dist_abs_now_gt: ', dist_abs_now_gt)
               # print ('reward_dist_gt: ', reward_dist_gt)
               reward = reward_dist_gt
               reward_dist_log_gt = float(reward_dist_gt)

               #======角度奖励部分=======
               print ('angle_final_gt: ', angle_final_gt)
               print ('angle_final_last: ', angle_final_last)



          # # print ('current_pos_gt: ', current_pos_gt)
          # # # current_pos_gt_rot = deepcopy(M_th.agent.state.rotation)
          # # # print ('current_pos_gt_rot: ', current_pos_gt_rot)
          # dist_ = local_goal_gt - current_pos_gt
          # dist_ = dist_.astype(float)
          # dist_abs_now_gt = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
          # # print ('dist_gt: ', dist_abs_now_gt)
          # #计算GT坐标系下的相对距离的奖励
          # reward_dist_gt = (dist_gt_last - dist_abs_now_gt) * self.rewad_arg1 - 0.01
          # print ('dist_gt_last: ', dist_gt_last)
          # print ('dist_abs_now_gt: ', dist_abs_now_gt)
          # # print ('reward_dist_gt: ', reward_dist_gt)
          # reward = reward_dist_gt
          # reward_dist_log_gt = float(reward_dist_gt)

          # #GT坐标系下计算和局部目标点之间的相对角度
          # cur_oritation_gt = M_th.agent.state.rotation
          # cur_oritation_gt_ = np.array([cur_oritation_gt.w, cur_oritation_gt.x, cur_oritation_gt.y, cur_oritation_gt.z])
          # angle_gt_now = get_angle_gt(cur_oritation_gt_)
          # angle_tensor_tensor = torch.tensor(angle_gt_now, dtype=torch.float)
          # angle_tensor_gt = angle_tensor_tensor.to(device0)
          # angle_final_gt = angle_GT(local_goal_gt, current_pos_gt, angle_tensor_gt)  #角度

          # #======角度奖励部分=======
          # # print ('angle_final_gt: ', angle_final_gt)
          # # print ('angle_final_last: ', angle_final_last)

          # # if (angle_final_gt > 0) and (angle_final_last >= 0):
          # #      reward_angle = (angle_final_last - angle_final_gt) * self.rewad_arg2
          # # elif (angle_final_gt > 0) and (angle_final_last < 0):
          # #      if (torch.abs(angle_final_gt) == torch.abs(angle_final_last)):
          # #           reward_angle = 0
          # #      elif (torch.abs(angle_final_gt) < torch.abs(angle_final_last)):
          # #           reward_angle = -(angle_final_last + angle_final_gt) * self.rewad_arg2
          # #      else:
          # #           reward_angle = -(angle_final_last + angle_final_gt) * self.rewad_arg2
          # # elif (angle_final_gt == 0):
          # #      reward_angle = torch.abs(angle_final_last) * self.rewad_arg2
          # # elif (angle_final_gt < 0) and (angle_final_last >= 0):
          # #      if (torch.abs(angle_final_gt) == torch.abs(angle_final_last)):
          # #           reward_angle = 0
          # #      else:
          # #           reward_angle = (angle_final_last + angle_final_gt) * self.rewad_arg2
          # # elif (angle_final_gt < 0) and (angle_final_last < 0):
          # #      reward_angle = (angle_final_gt - angle_final_last) * self.rewad_arg2

          # # print ('reward_angle: ', reward_angle)
          # # reward += reward_angle

          # # # reward_angle_log = reward_angle.detach().cpu().numpy()
          # # # reward_angle_log = float(reward_angle_log)
          # #======角度奖励部分=======

          if dist_abs_now_gt < self.distance_local_goal:
               done = True
               # reward += 20  #50
               reward += 12  #没有角度奖励
          else:
               done = False

          #以local_goal_为局部目标
          #TP坐标系下计算相对距离
          # print ('type(local_goal_): ', type(local_goal_))
          # print ('type(cur_pos_tp): ', type(cur_pos_tp))
          local_goal_tp = deepcopy(cur_pos_tp)
          local_goal_tp[0] = local_goal_[0]*0.04-20
          local_goal_tp[2] = local_goal_[1]*0.04-20
          local_goal_tp_array = np.array(local_goal_tp)
          dist_ = local_goal_tp_array - cur_pos_tp
          dist_ = dist_.astype(float)
          dist_abs_now_tp = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
          #记录TP坐标系下在地图上的相对距离和位置
          current_pos_tp_map2 = get_current_pose_tp(glovar.camera_trajectory)
          current_pos_tp_map = deepcopy(current_pos_tp_map2)
          current_pos_tp_map[0][1] = current_pos_tp_map2[0][0]
          current_pos_tp_map[0][0] = current_pos_tp_map2[0][1]
          #投影之后，TP坐标系下计算相对角度

          image_tensor, depth_tensor, mpt_tensor, angle_tensor_tp, obv_sts, map_data, mpt_num_now = M_th.get_states()

          current_pos_tp_ = [current_pos_tp_map[0][0], current_pos_tp_map[0][1]]
          angle_final = angle_TP(local_goal_, current_pos_tp_, angle_tensor_tp)
          angle_final_tp = angle_final
          angle_tp_now = angle_tensor_tp.detach().cpu().numpy()
          angle_tp_now = float(angle_tp_now)

          print ('reward total!!!!!!!!!!!!!!!!: ', reward)

          reward_mpt = 0
          reward_mpt_log = float(reward_mpt)

          return image_tensor, depth_tensor, mpt_tensor, angle_final_gt, reward, done, collided, reward_dist_log_gt, dist_abs_now_gt, angle_gt_now, dist_abs_now_tp, angle_final_tp, angle_tp_now, mpt_num_now, reward_mpt_log



     def _init_hyperparameters(self, hyperparameters):

          # Change any default values to custom values for specified hyperparameters
          for param, val in hyperparameters.items():
               exec('self.' + param + ' = ' + str(val))


class M_thread(threading.Thread):
     def __init__(self, threadID, name, counter, agent, sim_pathfinder, vel_control, time_step, sim_step_filter, model, writer):
          threading.Thread.__init__(self)
          self.threadID = threadID
          self.name = name
          self.counter = counter
          self.agent = agent
          self.sim_pathfinder = sim_pathfinder
          self.mp_gt = None
          self.vel_control = vel_control
          self.time_step = time_step
          self.sim_step_filter = sim_step_filter
          self.model = model
          self.R_ini = None
          self.t_ini = None


          self.writer = writer
          # self.total_t = total_t
          # self.total_update_epoch = total_update_epoch

          self._stop_event = threading.Event()
     
     def stop(self):
          self._stop_event.set()

     def get_states(self):
          time.sleep(0.6)

          threadLock.acquire()
          map_tp = get_mp_tp(glovar.map_data_)
          # current_pos_tp = get_current_pose_tp(glovar.camera_trajectory)
          current_pos_r, current_pos_t = get_current_pose(glovar.camera_trajectory)
          obv_mpt, obv_sts = get_orb_sts_mpt(glovar.orb_sts_mpt)

          map_data =  deepcopy(glovar.map_data_['map_data'])

          obv_image = deepcopy(glovar.image)
          obv_depth = deepcopy(glovar.depth)
          threadLock.release()

          dim = (320, 240)

          image_resize = cv2.resize(obv_image, dim, interpolation=cv2.INTER_AREA)

          image_resize = image_resize.astype(np.float32)/255.0
          image_tensor = torch.from_numpy(image_resize)
          indices = torch.LongTensor([0, 1, 2])
          image_tensor = torch.index_select(image_tensor, 2, indices)
          depth_resize = cv2.resize(obv_depth, dim, interpolation=cv2.INTER_AREA)

          depth_resize = depth_resize.astype(np.float32)/10.0  #8.0
          depth_tensor = torch.from_numpy(depth_resize)
          depth_tensor = depth_tensor.unsqueeze(2)

          # if glovar.obv['orb_sts_mpt']['sts'] == 2:
          if obv_sts == 2:
               # glovar.obv['orb_sts_mpt']['local_map'] = get_local_map(map_tp, current_pos_t)
               # glovar.obv['orb_sts_mpt']['angle'] = get_angle(current_pos_r)
               obv_angle = get_angle(current_pos_r)
               glovar.angle = deepcopy(obv_angle)
               # print ('glovar.obv[orb_sts_mpt][angle]', glovar.obv['orb_sts_mpt']['angle']) #-0.062424292770504645
               # print ('type(glovar.obv[orb_sts_mpt][angle])', type(glovar.obv['orb_sts_mpt']['angle'])) #<class 'numpy.float64'>

               if obv_mpt is None:
                    print ('obv_mpt is None')
                    print ('obv_mpt: ', obv_mpt)
                    print ('glovar.obv[orb_sts_mpt][sts]: ', glovar.obv['orb_sts_mpt']['sts'])
                    time.sleep(1.0)
                    # obv_mpt = deepcopy(glovar.obv['orb_sts_mpt']['mpt'])
                    # obv_mpt, glovar.obv['orb_sts_mpt']['sts'] = get_orb_sts_mpt(glovar.orb_sts_mpt)
                    # print ('glovar.obv[orb_sts_mpt][sts]', glovar.obv['orb_sts_mpt']['sts'])
               # elif obv_mpt is tuple:
               if isinstance(obv_mpt, tuple):
                    # obv_mpt = np.array(obv_mpt)
                    # time.sleep(1.0)
                    # obv_mpt = deepcopy(glovar.obv['orb_sts_mpt']['mpt'])
                    # print ('type(glovar.obv[orb_sts_mpt][mpt]', type(glovar.obv['orb_sts_mpt']['mpt']))
                    # print ('glovar.obv[orb_sts_mpt][mpt]', glovar.obv['orb_sts_mpt']['mpt'])
                    print ('obv_mpt: ', obv_mpt)
                    glovar.nan = 1
               # print ('type(obv_mpt): ', type(obv_mpt))
               # print ('obv_mpt.dtype: ', obv_mpt.dtype)
               # obv_mpt = np.array(obv_mpt)

               #统计特征点的个数
               # print ('obv_mpt.shape: ', obv_mpt.shape)  #(512, 512)
               obv_mpt_one = obv_mpt==1
               obv_mpt_num = obv_mpt[obv_mpt_one]
               # print ('obv_mpt_num.size: ', obv_mpt_num.size)
               obv_mpt_num_size = obv_mpt_num.size
               # print ('type(a): ', type(a))


               obv_mpt_tensor = torch.from_numpy(obv_mpt).float().unsqueeze(2).unsqueeze(3)
               obv_mpt_tensor = obv_mpt_tensor.transpose(0, 2)
               obv_mpt_tensor = obv_mpt_tensor.transpose(1, 3)
               out = nn.MaxPool2d(kernel_size=2, stride=2)
               obv_mpt_tensor_res = out.forward(obv_mpt_tensor)
               mpt_tensor = obv_mpt_tensor_res.squeeze(0).squeeze(0).unsqueeze(2)


               # angle_tensor = torch.tensor(glovar.obv['orb_sts_mpt']['angle'])
          else:

               mpt_tensor = torch.zeros([240, 320, 1])
               obv_mpt_num_size = 0.0
               # glovar.obv['orb_sts_mpt']['angle'] = 0.0  #为什么？？？
               obv_angle = 0.0
               print('no tracking')
               # angle_tensor = glovar.obv['orb_sts_mpt']['angle']

          angle = obv_angle
          # print ('angle: ', angle)

          # print ('type(angle): ', type(angle)) 
          angle_tensor_np = np.array(angle)  # b为numpy数据类型
          # angle_tensor_np = angle_tensor_np.float()
          # angle_tensor_np = angle_tensor_np + 180.0
          # angle_tensor_np = angle_tensor_np / 360.0
          angle_tensor_np = np.array(angle_tensor_np)  # b为numpy数据类型
          # angle_tensor_tensor = torch.from_numpy(angle_tensor_np)  # c为CPU的tensor
          angle_tensor_tensor = torch.tensor(angle_tensor_np, dtype=torch.float)
          angle_tensor = angle_tensor_tensor.to(device0)
          angle_tensor = -angle_tensor
          # print ('angle_tensor: ', angle_tensor) 
          # print ('type(angle_tensor): ', type(angle_tensor)) 

          
          # print ('image_tensor.shape0: ', image_tensor.shape)
          image_tensor = image_tensor.float()
          image_tensor = torch.transpose(image_tensor, 0, 2)
          # image_tensor = torch.transpose(image_tensor, 1, 2)  #加上的
          image_tensor = image_tensor.unsqueeze(0).to(device0)
          depth_tensor = depth_tensor.float()
          depth_tensor = torch.transpose(depth_tensor, 0, 2)
          depth_tensor = depth_tensor.unsqueeze(0)
          mpt_tensor = mpt_tensor.float()
          # mpt_tensor = torch.transpose(mpt_tensor, 0, 2)
          mpt_tensor = mpt_tensor.unsqueeze(0)

          #将depth_tensor和mpt_tensor变成3通道的
          n, c, h, w = image_tensor.size()        # batch_size, channels, height, weight
          depth_tensor = depth_tensor.view(n,h,w,1).repeat(1,1,1,c)
          depth_tensor = torch.transpose(depth_tensor, 1, 3)
          depth_tensor = torch.transpose(depth_tensor, 2, 3)  #加上的
          # depth_tensor = torch.transpose(depth_tensor, 0, 2)
          # depth_tensor = torch.transpose(depth_tensor, 1, 3)
          mpt_tensor = mpt_tensor.view(n,h,w,1).repeat(1,1,1,c)
          mpt_tensor = torch.transpose(mpt_tensor, 1, 3)
          mpt_tensor = torch.transpose(mpt_tensor, 2, 3)  #加上的
          depth_tensor = depth_tensor.to(device0)
          mpt_tensor = mpt_tensor.to(device0)

          return image_tensor, depth_tensor, mpt_tensor, angle_tensor, obv_sts, map_data, obv_mpt_num_size

     def run(self):
          q = self.agent.state.rotation
          t_ini = self.agent.state.position
          T = q_T(q, t_ini)
          self.R_ini = T.rotation
          self.t_ini = T.position
          self.model.learn(self, 20000, self.writer)


