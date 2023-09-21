import sys
import glovar

# sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
# sys.path.append('/home/jin/SLAM/ORB_SLAM_rgbd_loop_0410/ORB_SLAM_rgbd_loop/Examples/ROS/devel/lib/python2.7/dist-packages/')

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/media/vision/data/wang/ORB_SLAM_rgbd_loop_0410/ORB_SLAM_rgbd_loop/Examples/ROS/devel/lib/python2.7/dist-packages/')

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ros_Image
import habitat_sim.agent
from ros_thread import *
from image_pub import *
from sim_setting import *
from M_thread import *
from mapper import *
from orb_slam2_ros.srv import *
from orb_reset import *

# depth

# reset_flag = reset_flag
# gibision_dir = '/home/jin/RL-code/orbslam_sim/data/gibson_habitat/gibson'
gibision_dir = '/media/vision/data/wang/datasets/gibson_habitat/gibson_new'


class sim_env:
    def __init__(self, sim_=None, agent=None, vel_control=None, time_step=None, sence_=None):
        self.sim_ = sim_
        self.agent = agent
        self.vel_control = vel_control
        self.time_step = time_step
        self.resized_rgb_img = rospy.Publisher('camera/rgb/image_color', ros_Image, queue_size=10)
        self.resized_depth_img = rospy.Publisher('camera/depth/image', ros_Image, queue_size=10)
        self.meters_per_pixel = 0.025
        self.sence = sence_
        # self.reset_flag = reset_flag

    def env_make(self, emt):
        # print ('emt[scene_id]: ', emt['scene_id'])
        # print ('scene: ', emt['scene_id'].split('/')[-1].split('.')[0])
        scene_path = gibision_dir + '/' + emt['scene_id'].split('/')[-1]
        nav_path = gibision_dir + '/' + emt['scene_id'].split('/')[-1].split('.')[0] + '.navmesh'
        print ('scene_path: ', scene_path)
        print ('nav_path: ', nav_path)
        sim_settings["scene"] = scene_path
        cfg_ = make_cfg(sim_settings)
        self.sim_ = habitat_sim.Simulator(cfg_)
        self.sim_.seed(sim_settings["seed"])
        state = habitat_sim.agent.AgentState()
        state.position = np.array(emt['start_position'])
        state.rotation = np.array(emt['start_rotation'])
        agent: object = self.sim_.initialize_agent(sim_settings["default_agent"], state)
        self.agent = agent
        self.sim_.pathfinder.load_nav_mesh(nav_path)
        bounds = self.sim_.pathfinder.get_bounds()
        print ('bounds: ', bounds[0])
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.time_step = 0.033

    def env_start(self):
        set_glovar_var_ini()
        bridge = CvBridge()
        while not rospy.is_shutdown():
            obsv = self.sim_.get_sensor_observations()
            glovar.image = obsv["color_sensor"]
            glovar.depth = obsv["depth_sensor"]
            rgb_img = Image.fromarray(glovar.image.astype(np.uint8), mode="RGBA")
            rgb_img = rgb_img.convert('RGB')
            rgb_img_arr = np.array(rgb_img)
            cv_img = bridge.cv2_to_imgmsg(rgb_img_arr, "passthrough")
            self.resized_rgb_img.publish(cv_img)
            depth_int = (glovar.depth * 5000).astype(np.uint16)
            depth_img_arr = np.array(depth_int)
            cv_img_depth = bridge.cv2_to_imgmsg(depth_img_arr, "passthrough")
            self.resized_depth_img.publish(cv_img_depth)
            time.sleep(0.023)
            # print('reset_fflag', reset_flag)
            if glovar.sim_reset:
                break
            if glovar.nan:
                break

        
        

    def env_reset(self, emt):
        state = habitat_sim.agent.AgentState()
        state.position = np.array(emt['start_position'])
        state.rotation = np.array(emt['start_rotation'])
        self.agent.initial_state = state
        self.sim_.reset()
        # state = self.agent.state
        # state.position = np.array(emt['start_position'])
        # state.rotation = np.array(emt['start_rotation'])
        # self.agent.set_state(state, infer_sensor_states=False)
        orb_slam_reset()
        glovar.sim_reset = 0
        self.env_start()

    def sence_change(self, emt, model, writer):
        time.sleep(1.0)
        self.sim_.close()
        orb_slam_reset()
        glovar.sim_reset = 0
        # print ('start env_make!!!')
        self.env_make(emt)
        # print ('start env_thread!!!')
        # self.env_thread(model, id_x, filename, epid)
        self.env_thread(model, writer)
        # print ('start env_start!!!')
        self.env_start()


    def env_thread(self, model, writer):
        time.sleep(0.5)
        glovar.thread = []

        if (glovar.start_flag == 1):
            try:
                thread1 = rosthread(1, "Thread1", 10)
                thread1.daemon = True
                thread1.start()
                glovar.thread.append(thread1)
            except rospy.ROSInterruptException:
                rospy.loginfo('map_data terminated.')
            print('map is received')


        try:
            
            thread2 = M_thread(2, "Thread2", 10, self.agent, self.sim_.pathfinder, self.vel_control, self.time_step,
                               self.sim_.step_filter, model, writer)
            thread2.daemon = True
            thread2.start()
            glovar.thread.append(thread2)
            #thread1.join()
            #thread2.join()
        except rospy.ROSInterruptException:
            rospy.loginfo('map_data terminated.')


if __name__ == "__main__":
    # scene = choose_rand_sence()
    scene = '/home/zsk/script/habitat/orbslam_sim/data/25.glb'
    rospy.init_node('test_map', anonymous=False, disable_signals=True)
    env = sim_env()
    env.env_make(scene)
    env.env_thread()
    env.env_start()
    for i in range(10):
        time.sleep(1.0)
        if glovar.sim_reset == 1:
            env.env_reset()
        elif glovar.sence_change == 1:
            env.sence_change()
        else:
            pass
    print('程序结束')
