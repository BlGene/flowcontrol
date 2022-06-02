import os
import json
import shutil
import unittest
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_io.recorder.simple_recorder import load_rec_list
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.demo.demo_trajectory_utils import split_recording
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


class ShapeSorting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)

        cls.orn_options = dict(
            #rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
            #rZ=R.from_euler("xyz", (0, 0, 20), degrees=True).as_quat(),
            rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
            #rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
            #rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
            )

        cls.save_dir_template = "./tmp_test/shape_sorting"

    # TODO(max): sample_params False, but chaning seed still changes values.
    def test_01_record(self):
        seed = 3

        for name, orn in self.orn_options.items():
            env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=200, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info={"trapeze_pose": [[0.043, -0.60, 0.140], orn]},
                              seed=seed)

            save_dir = self.save_dir_template + f"_{name}"
            if os.path.isdir(save_dir):
                # lsof file if there are NSF issues.
                shutil.rmtree(save_dir)
            record_sim(env, save_dir)
            break

    def test_02_segment(self):
        # segment the demonstration

        # Convert notebook to script
        convert_cmd = "jupyter nbconvert --to script ./demo/Demonstration_Viewer.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd)

        for name, orn in self.orn_options.items():
            # Run generated script
            save_dir = self.save_dir_template + f"_{name}"
            segment_cmd = f"python ./demo/Demonstration_Viewer.py {save_dir}"
            subprocess.run(segment_cmd.split(), check=True)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo/Demonstration_Viewer.py")

    def test_03_servo(self):
        seed = 3
        control_config = dict(mode="pointcloud", threshold=0.41)

        for name, orn in self.orn_options.items():
            save_dir = self.save_dir_template + f"_{name}"

            servo_module = ServoingModule(save_dir,
                                          episode_num=0,
                                          control_config=control_config,
                                          plot=True, save_dir=None)


            env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=500, control='relative',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info={"trapeze_pose": [[0.043, -0.60, 0.140], orn]},
                              seed=seed
                              )

            state, reward, done, info = evaluate_control(env, servo_module)
            self.assertEqual(reward, 1.0)

    """
    def test_05_split(self):
        name = "rZ"
        save_dir = self.save_dir + f"_{name}"
        split_recording(save_dir)
    """

if __name__ == '__main__':
    unittest.main()
