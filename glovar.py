import numpy as np

sim_reset = 0
#sence_change = 0
camera_trajectory = {'data': []}
map_data_ = {'map_data': []}
orb_sts_mpt = {'mpt': [], 'sts': 0}
image = np.zeros((480, 640))
#image = np.zeros((512, 512, 4))
depth = np.zeros((480, 640, 4))
#depth = np.zeros((512, 512))
global_goal_gt = None
angle = None
local_map = []
local_goal_point = []
# obv = {'orb_sts_mpt': orb_sts_mpt, 'image': image, 'depth': depth, 'local_map': local_map,
#        'angle': angle}
# obv = {'orb_sts_mpt': None, 'image': None, 'depth': None, 'local_map': None,
#        'angle': None}
local_goal_point = []
change_flag = 0
thread = []
log_local_success = []
log_global_success = []
log_global_num = []
log_global_tr_num = []
log_global_collision_num = []
log_global_success_num = []
log_id_x = []
log_epid = []

total_update_epoch = 0
total_t = 0
nan = 0

random = 0

train_num = 0
train_local_num = 0
train_global_num = 0
local_success_num = 0
global_success_num = 0
global_tr_num = 0
global_collision_num = 0

episode_id = 0
scene_id = 0


# fail_pos = [[-2.58680009841919, 0.108418226242065, 1.52841556072235]]
# fail_rot = [[0.0, -0.975159525871277, 0.0, -0.221503913402557]]

fail_pos = []
fail_rot = []
fail_id_x = []
fail_epid = []

pos_tem = []
rot_tem = []


local_num = 0

start_flag = 1

test = 0

success = 0
dis = 0
agent_move = 0


avg_reward = 0
reward_sum = 0
reward_step = 0

len = []
avg = []
step = []

local_steps = 0
image_tensor_log = []
depth_tensor_log = []
mpt_tensor_log = []

