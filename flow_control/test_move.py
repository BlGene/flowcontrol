"""
Test functional beahviour through built-in policies.
"""
import os
import time
import logging
import unittest
from pdb import set_trace
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule, T_TCP_CAM
from gym_grasping.utils import state2matrix

is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"

R_XZ2 = np.eye(4)
R_XZ2[:3, :3] = R.from_euler('xyz', (90, 0, 180), degrees=True).as_matrix()


def make_demo_dict(env, base_state, base_info, base_action):
    """
    create keep dict with info from state
    """
    env_info = env.get_info()
    rgb = base_state[np.newaxis, :]
    depth = base_info["depth"][np.newaxis, :]
    mask = base_info["seg_mask"][np.newaxis, :] == 2
    state = base_info["robot_state_full"][np.newaxis, :]
    keep_dict = {0: None}

    actions = np.array(base_action)[np.newaxis, :]

    demo_dict = dict(env_info=env_info,
                     rgb=rgb,
                     depth=depth,
                     mask=mask,
                     state=state,
                     keep_dict=keep_dict,
                     actions=actions)
    return demo_dict


class Move_absolute(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """

    def test_gripper(self):
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        base_state, _, _, base_info = env.step(None)
        base_tcp_pose = env.robot.get_tcp_pose()
        base_tcp_pos = env.robot.get_tcp_pos()
        cam_mat = env.camera.get_cam_mat()
        cam_mat = cam_mat @ R_XZ2

        env.p.addUserDebugLine([0, 0, 0], base_tcp_pose[:3, 3], lineColorRGB=[1, 0, 0],
                               lineWidth=2, physicsClientId=0)
        # green line to object
        env.p.addUserDebugLine([0, 0, 0], cam_mat[:3, 3], lineColorRGB=[0, 1, 0],
                               lineWidth=2, physicsClientId=0)


        print("cam_mat\n", cam_mat.round(3))
        print("tcp_pose\n", base_tcp_pose.round(3))
        guess2 = np.linalg.inv(cam_mat) @ base_tcp_pose
        print(guess2.round(3))
        print(guess2[:3, 3])

        pt =  cam_mat @ guess2
        env.p.addUserDebugLine(cam_mat[:3, 3], pt[:3, 3], lineColorRGB=[0, 0, 1],
                               lineWidth=2, physicsClientId=0)

        tcp_angles = env.robot.get_tcp_angles()

        base_action = [*base_tcp_pos, tcp_angles[2], 1]
        #control = env.robot.get_control("absolute", min_iter=24)

        """
        for i in range(10):
            state2, _, _, _ = env.step(action, control)
            tcp_goal = tcp_pos.copy()
            tcp_goal[]
            action = [*tcp_pos, tcp_angles[2], 1]
            control = env.robot.get_control("absolute", min_iter=24)
        """

        delta = 0.04
        data = []
        for i in (0, 1, 2):
            for j in (1, -1):
                target_pose = list(base_tcp_pos)
                target_pose[i] += j * delta

                action = [*target_pose, tcp_angles[2], 1]
                control = env.robot.get_control("absolute") #, min_iter=24)
                state2, _, _, info = env.step(action, control)
                tcp_pose = env.robot.get_tcp_pose()
                data.append(dict(action=action, state=state2, info=info,
                                 pose=tcp_pose))


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
                                      camera_calibration=env.camera.calibration,
                                      plot=True, save_dir=None)
        print("loaded servo module")

        for i in range(len(data)):
            rgb = data[i]["state"]
            state = data[i]["info"]["robot_state_full"]
            depth = data[i]["info"]["depth"]
            action, done, info = servo_module.step(rgb, state, depth)

            align_trf = action[0]
            move_w = data[i]["pose"] @ np.linalg.inv(base_tcp_pose)

            est_t = np.linalg.inv(guess2) @ align_trf @ guess2
            est_w = np.linalg.inv(base_tcp_pose) @ est_t @ base_tcp_pose


            align_trf = action[0]

            print(move_w.round(3))
            print()
            print(est_t.round(3))
            set_trace()



        """
        success_count = 0
        start_time = time.time()
        env_done = False
        for iteration in range(500):
            action, policy_done = env._task.policy(env)
            _, reward, env_done, _ = env.step(action)

            if env_done and reward > 0:
                success_count += 1
                break

        end_time = time.time()

        time_target_s = 3.0
        if env._renderer == "debug":
            time_target_s *= 2

        self.assertGreater(success_count, 0)
        self.assertLess(end_time-start_time, time_target_s)
        """

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
