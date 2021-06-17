"""
Test functional behaviour through built-in policies.
"""
import os
import math
import logging
import unittest
import numpy as np

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule
is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"


class MoveThenServo(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """
    def test_absolute(self):
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        # record base frame
        base_state, _, _, base_info = env.step(None)
        tcp_base = env.robot.get_tcp_pose()
        tcp_angles = env.robot.get_tcp_angles()
        # relative transformations
        cam_base = env.camera.get_cam_mat()
        # cam_tcp = np.linalg.inv(cam_base) @ tcp_base

        data = []

        target_pose, control = next(get_target_poses(env, tcp_base))

        # go to state
        action = [*target_pose, tcp_angles[2], 1]
        state2, _, _, info = env.step(action, control)
        # and collect data
        tcp_pose = env.robot.get_tcp_pose()
        cam_pose = env.camera.get_cam_mat()
        data.append(dict(action=action, state=state2, info=info,
                         pose=tcp_pose, cam=cam_pose))

        # initialize servo module
        base_action = [*tcp_base[:3, 3], tcp_angles[2], 1]
        demo_dict = make_demo_dict(env, base_state, base_info, base_action)
        control_config = dict(mode="pointcloud-abs",
                              gain_xy=50,
                              gain_z=100,
                              gain_r=15,
                              threshold=0.40)

        servo_module = ServoingModule(demo_dict,
                                      episode_num=0,
                                      start_index=0,
                                      control_config=control_config,
                                      plot=True, save_dir=None)

        max_steps = 1000
        servo_action = None
        servo_control = None  # means default
        done = False
        for counter in range(max_steps):
            # environment stepping
            if done:
                break

            state, reward, done, info = env.step(servo_action, servo_control)
            # compute action
            if isinstance(env, RobotSimEnv):
                obs_image = state  # TODO(max): fix API change between sim and robot
            else:
                obs_image = info['rgb_unscaled']
            ee_pos = info['robot_state_full'][:8]  # take three position values
            servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
            servo_action, servo_done, servo_queue = servo_res

            if servo_module.config.mode == "pointcloud-abs":
                trf_est, grip_action = servo_action
                # robot_pose = env.robot.get_tcp_pose()
                goal_pos = cam_base @ trf_est @ np.linalg.inv(cam_base) @ tcp_base
                goal_pos = goal_pos[:3, 3]
                print(goal_pos)

                # env.p.removeAllUserDebugItems()
                # env.p.addUserDebugLine([0, 0, 0], robot_pose[:3, 3], lineColorRGB=[1, 0, 0],
                #                        lineWidth=2, physicsClientId=0)
                # env.p.addUserDebugLine([0, 0, 0], goal_pos, lineColorRGB=[0, 1, 0],
                #                        lineWidth=2, physicsClientId=0)

                goal_angle = math.pi / 4
                servo_action = goal_pos.tolist() + [goal_angle, grip_action]
                servo_control = env.robot.get_control("absolute")
                # set_trace()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
