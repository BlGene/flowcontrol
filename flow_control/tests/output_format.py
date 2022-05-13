"""
Test functional beahviour through built-in policies.
"""
import os
import math
import logging
import unittest
import numpy as np

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from gym_grasping.envs.iiwa_env import IIWAEnv


is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"


class OutputFormat(unittest.TestCase):
    def check_format(self, state, info):
        self.assertTrue(len(state.shape) == 3)
        self.assertTrue(state.shape[2] == 3)

        robot_state_full = info["robot_state_full"]
        self.assertTrue(len(robot_state_full.shape) == 1)
        self.assertTrue(robot_state_full.dtype == np.float32)

        depth = state["depth_gripper"]
        self.assertTrue(len(depth.shape) == 2)
        print(depth.dtype)
        self.assertTrue(depth.dtype == np.float32)

        if "seg_mask" not in info:
            return

        seg_mask = info["seg_mask"]
        self.assertTrue(len(seg_mask.shape) == 2)
        self.assertTrue(seg_mask.dtype == np.int32)

    def test_output_format(self, is_sim=False):
        if is_sim:
            env = RobotSimEnv(task="flow_calib", robot="kuka",
                              obs_type=obs_type, renderer=renderer,
                              act_type='continuous', control="absolute",
                              max_steps=600, initial_pose="close",
                              img_size=(256, 256),
                              param_randomize=False)
        else:
            env = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced', dv=0.01,
                          drot=0.04, joint_vel=0.05,  # trajectory_type='lin',
                          gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True, safety_stop=True,
                          dof='5dof',
                          reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2),
                          obs_dict=False)

        state = env.reset()
        info = env.get_obs_info()
        self.check_format(state, info)

        state, _, _, info = env.step(None)
        self.check_format(state, info)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
