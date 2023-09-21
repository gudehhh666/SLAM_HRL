"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gzip
import json
import glob
import random
from PPO.arguments import get_args
from PPO.network_att import Actor_Net, Critic_Net
# from network.network_DAC import OptionGaussianActorCriticNet
# from network.network_SAC import OptionGaussianActorCriticNet
# from network.network_SAC import Actor
from network.network_TD3 import *
from PPO.eval_policy import eval_policy
from habitat_env import *
from torch.utils.tensorboard import SummaryWriter
import xlrd

import inspect
import ctypes



def read_excel(scene, num):
    # workbook = xlrd.open_workbook(r'/home/jin/RL-code/orbslam_sim_old/perfect/train_atLoc/four_act/test/test_gibson_val_dis_6_19_ddppo.xls')
    
    workbook = xlrd.open_workbook(r'/media/vision/data/wang/datasets/test_gibson/test_gibson_val_dis_6_19_ddppo.xls')
    sheet2_name= workbook.sheet_names()[0]
    sheet2 = workbook.sheet_by_name(sheet2_name)
    dis = 0
    # print ('scene ', scene[7:])
    # print ('num ', num)
    # print (sheet2.row(6)[0].value[27:])
    for i in range(1, 995, 1):
        if (sheet2.row(i)[0].value[27:] == scene[7:]) and (sheet2.row(i)[1].value == num):
            dis = sheet2.row(i)[-2].value
    return dis



if __name__ == '__main__':
    hyperparameters = {

        'steps_per_local_goal': 12, #12
        # 'timesteps_per_batch': 2048,
        # 'max_timesteps_per_episode': 200,
        'gamma': 0.8,
        'n_updates_per_iteration': 4,  #6
        # 'lr': 0.05, #3e-4,
        'lr_actor': 3e-4, #3e-4,
        'lr_critic': 3e-3, #3e-4,
        'clip': 0.2,  #0.2
        # 'render': True,
        # 'render_every_i': 10
    }
    args = get_args()  # Parse arguments from command line
    rospy.init_node('test_map', anonymous=False, disable_signals=True)
    # point_goal_task_train = '/home/jin/RL-code/orbslam_sim/data/pointnav_gibson_v1/train/content/*.json.gz'
    # point_goal_task_train = '/home/jin/habitat/habitat-challenge/habitat-challenge-data/data/datasets/pointnav/gibson/v2/val/content/*.json.gz'
    # scenes_dir = r'/home/jin/RL-code/orbslam_sim/data/gibson_habitat/gibson/'
    
    point_goal_task_train = '/media/vision/data/wang/datasets/content/*.json.gz'
    scenes_dir = r'/media/vision/data/wang/datasets/gibson_habitat/gibson_new'
    
    time_per_sence = 10
    point_goal_task_files = glob.glob(point_goal_task_train)
    env = sim_env()
    
    # fail_path = '/home/jin/RL-code/orbslam_sim_old/perfect/log/date6_24_test_4_/fail_492.xls'
    # total_update_epoch = 0
    # total_t = 0
    print ('args.test: ', args.test)
    glovar.test = args.test

    if args.mode == 'train':
        # SAC_model = args.SAC_model
        TD3_model = args.actor
        model = TD3(policy_class= HRLTD3(2), **hyperparameters)
        if (TD3_model != ''):
            print(f"Loading in {TD3_model}", flush=True)
            model.actor.load_state_dict(torch.load(TD3_model))
            model.critic.load_state_dict(torch.load(args.critic))
            model.actor_target.load_state_dict(torch.load(args.actor_target))
            model.critic_target.load_state_dict(torch.load(args.critic_target))
            print(f"Successfully loaded.", flush=True)
        else:
            print(f"Training from scratch.", flush=True)
        
        start_flag = 1

        write_log = "/media/vision/data/wang/codes/TD3_HRL/my_experiment/date4_11_3_" + str(glovar.test)
        writer = SummaryWriter(log_dir=write_log)

        glovar.random = 0

        # for i in range(len(point_goal_task_files)):  # 训练场景数量
        for i in range(100):  # 训练场景数量
            # print (point_goal_task_files[i])
            print (point_goal_task_files)
            # glovar.id_x = random.randint(0, len(point_goal_task_files) - 1)
            # print (glovar.id_x)
            glovar.id_x = random.randint(0, len(point_goal_task_files) - 1)
            print (glovar.id_x)  #7

            # aa = 1
            if (glovar.id_x == 6) or (glovar.id_x == 7) or (glovar.id_x == 9) or (glovar.id_x == 13) or (glovar.id_x == 2):
            
                aa = 1
            else:
                aa = 0

            # #===指定场景===
            # # glovar.id_x = 7  #7
            # #顺序选择场景 
            # # (i == 0): Edgemere; (i == 1):Pablo; (i == 3):Greigsville; (i == 6):Ribera; (i == 7):Denmark;
            # #  (i == 2):Scioto; (i == 9):Swormville;
            # # if (i == 6) or (i == 7) or (i == 0) or (i == 1) or (i == 2):
            # if (glovar.id_x == 7):  
            #     aa = 1
            # else:
            #     aa = 0
            # #===指定场景===

            with gzip.open(point_goal_task_files[glovar.id_x], "rt") as f:
                deserialized = json.loads(f.read())
                num_task_goal = deserialized['episodes'][-1]['episode_id']
                for k in range(time_per_sence):
                    glovar.epid = random.randint(0, int(num_task_goal))
                    print ('glovar.epid: ', glovar.epid)  #17
                    # glovar.epid = 17   #定义场景，定义17(对应7场景)
                    
                    for j, emt in enumerate(deserialized['episodes']):
                        if (j == glovar.epid):       #if (i == 2):
                            if (int(emt['episode_id']) >= 0) and (aa == 1):     #if (int(emt['episode_id']) >= 19):
                                glovar.global_goal_gt = emt['goals'][0]['position']
                                print ('emt[episode_id]: ', emt['episode_id'])  #17
                                print ('emt[scene_id]: ', emt['scene_id'])   #gibson/Denmark.glb
                                glovar.episode_id = emt['episode_id']
                                glovar.scene_id = emt['scene_id']
                                glovar.dis = read_excel(emt['scene_id'], emt['episode_id'])
                                print ('glovar.dis: ', glovar.dis)
                                glovar.success = 0
                                glovar.agent_move = 0
                                glovar.train_num = 0
                                if (glovar.dis == '') or (glovar.dis == 0):
                                    continue
                                else:
                                    if start_flag == 1:
                                        env.env_make(emt)
                                        print('emt_start', emt['start_position'])
                                        # env.env_thread(model, id_x, filename, epid)
                                        env.env_thread(model, writer)
                                        env.env_start()
                                        # time.sleep(2.0)
                                        start_flag = 0
                                    else:
                                        if glovar.change_flag == 0:  #没有换场景，只是换了起点
                                            env.env_reset(emt)
                                        else:
                                            glovar.thread[1].join()
                                            env.sence_change(emt, model, writer)
                                            glovar.change_flag = 0
            glovar.change_flag = 1


