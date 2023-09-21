"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

def get_args():
	"""
		Description:
		Parses arguments at command line.

		Parameters:
			None

		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--test', dest='test', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	# parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # /home/jin/RL-code/orbslam_sim_old/perfect/train_atLoc/safe_mode/model/date7_29_reward_1_/final/ppo_actor.pth
	# parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # /home/jin/RL-code/orbslam_sim_old/perfect/train_atLoc/safe_mode/model/date7_29_reward_1_/final/ppo_critic.pth
	# parser.add_argument('--SAC_model', dest='SAC_model', type=str, default='/home/jin/RL-code/orbslam_sim_old/perfect/option/SAC2/model/date4_7/train/SAC_actor.pth')
	# parser.add_argument('--SAC_model', dest='SAC_model', type=str, default='/home//RL-code/orbslam_sim_old/perfect/option/SAC2/model/date4_7_2/train/SAC_actor.pth')
	# parser.add_argument('--SAC_model', dest='SAC_model', type=str, default='/home/jin/RL-code/orbslam_sim_old/perfect/option/SAC2/model/date4_7_6/train/SAC_actor.pth')
	# parser.add_argument('--SAC_model', dest='SAC_model', type=str, default='')

	parser.add_argument('--actor', dest='actor', type=str, default='')
	parser.add_argument('--critic', dest='critic', type=str, default='')
	parser.add_argument('--actor_target', dest='actor_target', type=str, default='')
	parser.add_argument('--critic_target', dest='critic_target', type=str, default='')
	# parser.add_argument('--actor', dest='actor', type=str, default='/home/jin/RL-code/orbslam_sim_old/perfect/option/TD3/model/date4_11_2/train/TD3_actor.pth')
	# parser.add_argument('--critic', dest='critic', type=str, default='/home/jin/RL-code/orbslam_sim_old/perfect/option/TD3/model/date4_11_2/train/TD3_critic.pth')
	# parser.add_argument('--actor_target', dest='actor_target', type=str, default='/home/jin/RL-code/orbslam_sim_old/perfect/option/TD3/model/date4_11_2/train/TD3_actor_target.pth')
	# parser.add_argument('--critic_target', dest='critic_target', type=str, default='/home/jin/RL-code/orbslam_sim_old/perfect/option/TD3/model/date4_11_2/train/TD3_critic_target.pth')
	args = parser.parse_args()

	return args
