import os
import json
import shutil
import unittest
import subprocess

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


class TestRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)

        cls.save_dir = "./tmp_test/shape_sorting"
        cls.episode_num = 0

        if os.path.isdir(cls.save_dir):
            # lsof file if there are NSF issues.
            shutil.rmtree(cls.save_dir)

    # comment this out to keep ./tmp_test files
    # @classmethod
    # def tearDownClass(cls):
    #    try:
    #        shutil.rmtree(cls.save_dir)
    #    except OSError:
    #        # lsof file if there are NSF issues.
    #        pass


    # TODO(max): sample_params False, but chaning seed still changes values.
    def test_01_record(self):
        seed = 3
        orn_options = dict(
            rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
            rZ=R.from_euler("xyz", (0, 0, 15), degrees=True).as_quat(),
            rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
            rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
            rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat())

        for name, orn in orn_options.items():
            env = RobotSimEnv(task='shape_sorting', renderer='egl', act_type='continuous',
                              initial_pose='close', max_steps=200, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info={"trapeze_pose": [[-0.043, -0.644, 0.140], orn]},
                              seed=seed)

            record_sim(env, self.save_dir + f"_{name}")
            break


if __name__ == '__main__':
    unittest.main()
