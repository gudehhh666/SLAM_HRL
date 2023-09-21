import habitat_sim
import habitat_sim.agent
import time
from habitat_sim.utils import common as utils
import numpy as np
from copy import deepcopy
import glovar

def action(agent, ctrl):
    if ctrl == 0:
        ctr = "move_forward"
        discrete_action = agent.agent_config.action_space[ctr]
        if agent.controls.is_body_action(discrete_action.name):
            did_collide = agent.controls.action(agent.scene_node, discrete_action.name,
                                                discrete_action.actuation, apply_filter=True)
        else:
            for _, v in agent._sensors.items():
                habitat_sim.errors.assert_obj_valid(v)
                agent.controls.action(v.object, discrete_action.name, discrete_action.actuation,
                                      apply_filter=False)
    elif ctrl == 1:
        ctr = "turn_left"
        discrete_action = agent.agent_config.action_space[ctr]
        if agent.controls.is_body_action(discrete_action.name):
            did_collide = agent.controls.action(agent.scene_node, discrete_action.name,
                                                discrete_action.actuation, apply_filter=True)
        else:
            for _, v in agent._sensors.items():
                habitat_sim.errors.assert_obj_valid(v)
                agent.controls.action(v.object, discrete_action.name, discrete_action.actuation,
                                      apply_filter=False)
    elif ctrl == 2:
        ctr = "turn_right"
        discrete_action = agent.agent_config.action_space[ctr]
        if agent.controls.is_body_action(discrete_action.name):
            did_collide = agent.controls.action(agent.scene_node, discrete_action.name,
                                                discrete_action.actuation, apply_filter=True)
        else:
            for _, v in agent._sensors.items():
                habitat_sim.errors.assert_obj_valid(v)
                agent.controls.action(v.object, discrete_action.name, discrete_action.actuation,
                                      apply_filter=False)
    else:
        print("error action")


def actiopn_n(agent, ctrl, num):
    for i in range(num):
        action(agent, ctrl)
        time.sleep(0.1)


# 连续动作
def actiopn_vel(agent, ctrl, vel_control, time_step, sim_step_filter):
    # print ('ctrl: ', ctrl)
    # print ('ctrl.shape: ', ctrl.shape)
    if ctrl == 1:
        ctr = "move_forward"
        # vel_control.linear_velocity = np.array([0, 0, -0.5])
        # vel_control.angular_velocity = np.array([0, 0.1, 0])
        vel_control.linear_velocity = np.array([0, 0, -0.3])  #0.5
        vel_control.angular_velocity = np.array([0, 0.0001, 0])
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position)
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state)
        end_pos = sim_step_filter(
            previous_rigid_state.translation, target_rigid_state.translation)
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation)
        agent.set_state(agent_state)
        dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation).dot()
        dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation).dot()
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        return collided
    elif ctrl == 0:
        ctr = "turn_left"
        # vel_control.linear_velocity = np.array([0, 0, -0.5])
        # vel_control.angular_velocity = np.array([0, 0.5, 0])
        vel_control.linear_velocity = np.array([0, 0, -0.01])
        vel_control.angular_velocity = np.array([0, 0.7, 0])
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position)
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state)
        end_pos = sim_step_filter(
            previous_rigid_state.translation, target_rigid_state.translation)
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation)
        agent.set_state(agent_state)
        dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation).dot()
        dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation).dot()
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        return collided
    elif ctrl == 2:
        ctr = "turn_right"
        # vel_control.linear_velocity = np.array([0, 0, -0.5])
        # vel_control.angular_velocity = np.array([0, -0.5, 0])
        vel_control.linear_velocity = np.array([0, 0, -0.01])
        vel_control.angular_velocity = np.array([0, -0.7, 0])
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position)
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state)
        end_pos = sim_step_filter(
            previous_rigid_state.translation, target_rigid_state.translation)
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation)
        agent.set_state(agent_state)
        dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation).dot()
        dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation).dot()
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        return collided
    # elif ctrl == 3:
    #     ctr = "backward"
    #     # vel_control.linear_velocity = np.array([0, 0, -0.5])
    #     # vel_control.angular_velocity = np.array([0, 0.1, 0])
    #     vel_control.linear_velocity = np.array([0, 0, 0.3])
    #     vel_control.angular_velocity = np.array([0, 0.0001, 0])
    #     agent_state = agent.state
    #     previous_rigid_state = habitat_sim.RigidState(
    #         utils.quat_to_magnum(agent_state.rotation), agent_state.position)
    #     target_rigid_state = vel_control.integrate_transform(
    #         time_step, previous_rigid_state)
    #     end_pos = sim_step_filter(
    #         previous_rigid_state.translation, target_rigid_state.translation)
    #     agent_state.position = end_pos
    #     agent_state.rotation = utils.quat_from_magnum(
    #         target_rigid_state.rotation)
    #     agent.set_state(agent_state)
    #     dist_moved_before_filter = (
    #             target_rigid_state.translation - previous_rigid_state.translation).dot()
    #     dist_moved_after_filter = (
    #             end_pos - previous_rigid_state.translation).dot()
    #     EPS = 1e-5
    #     collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    #     return collided
    # else:
    #     print("error action")

    # time.sleep(0.1)

def euclidean_distance(position_a, position_b):
    dist_ = position_a - position_b
    dist_ = dist_.astype(float)
    dist_abs_now_gt = (dist_[0] ** 2 + dist_[2] ** 2) ** 0.5
    return dist_abs_now_gt


def actiopn_n_vel(agent, ctrl, vel_control, time_step, sim_step_filter, num=3):
    previous_pos_gt = deepcopy(agent.state.position)
    if (ctrl == 1):
        for i in range(8):  #8
            collided = actiopn_vel(agent, ctrl, vel_control, time_step, sim_step_filter)
            if collided:
                print('collision happen!!!!!!!!collision happen!!!!!!!!')
                break
            time.sleep(0.1)
    # elif (ctrl == 3):
    #     for i in range(3):  #8
    #         collided = actiopn_vel(agent, ctrl, vel_control, time_step, sim_step_filter)
    #         if collided:
    #             print('collision happen!!!!!!!!collision happen!!!!!!!!')
    #             break
    #         time.sleep(0.1)
    else:
        for i in range(5):   #5
            collided = actiopn_vel(agent, ctrl, vel_control, time_step, sim_step_filter)
            if collided:
                print('collision happen!!!!!!!!collision happen!!!!!!!!')
                break
            time.sleep(0.1)

    current_pos_gt = deepcopy(agent.state.position)
    glovar.agent_move += euclidean_distance(previous_pos_gt, current_pos_gt)


    # for i in range(num):
    #     collided = actiopn_vel(agent, ctrl, vel_control, time_step, sim_step_filter)
    #     if collided:
    #         print('collision happen!!!!!!!!collision happen!!!!!!!!')
    #         break
    #     time.sleep(0.1)
    return collided



# 连续动作
def actiopn_vel_new(agent, ctrl, vel_control, time_step, sim_step_filter):
    for i in range(5):
        # vel_control.linear_velocity = np.array([0, 0, ctrl[0][0].detach().cpu().numpy()])  #0.5
        # vel_control.angular_velocity = np.array([0, ctrl[0][1].detach().cpu().numpy(), 0])
        vel_control.linear_velocity = np.array([0, 0, ctrl[0]])  #0.5
        vel_control.angular_velocity = np.array([0, ctrl[1], 0])
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position)
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state)
        end_pos = sim_step_filter(
            previous_rigid_state.translation, target_rigid_state.translation)
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation)
        agent.set_state(agent_state)
        dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation).dot()
        dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation).dot()
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        if collided:
            print('collision happen!!!!!!!!collision happen!!!!!!!!')
            break
        time.sleep(0.2)
    return collided














def actiopn_n_vel_start(agent, ctrl, vel_control, time_step, sim_step_filter, num=3):
    for i in range(num):
        collided = actiopn_vel(agent, ctrl, vel_control, time_step, sim_step_filter)
        if collided:
            print('collision happen!!!!!!!!collision happen!!!!!!!!')
            break
        time.sleep(0.1)
    return collided







