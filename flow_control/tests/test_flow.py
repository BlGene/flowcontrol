"""
Test servoing for pick_n_place task.
"""
import os
import json
import shutil
import unittest
import subprocess

from gym_grasping.envs.robot_sim_env import RobotSimEnv

from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


class TestFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)
        cls.save_dir = "./tmp_test/pick_n_place"

    # comment this out to keep ./tmp_test files
    # @classmethod
    # def tearDownClass(cls):
    #    try:
    #        shutil.rmtree(cls.save_dir)
    #    except OSError:
    #        # lsof file if there are NSF issues.
    #        pass

    def test_01_record(self):
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)

        env = RobotSimEnv(task='pick_n_place', renderer='egl', act_type='continuous',
                          initial_pose='close', max_steps=200, control='absolute-full',
                          img_size=(256, 256), sample_params=False)

        record_sim(env, self.save_dir)

    def test_02_segment(self):
        # segment the demonstration

        # Save configuration
        conf_objects = dict(
            blue_block=[{'name': 'color', 'color': [0, 0, 1], 'threshold': 0.65},
                        {'name': 'center'}],
            red_nest=[{'name': 'color', 'color': [1, 0, 0], 'threshold': 0.9},
                      {'name': 'center'}])
        conf_sequence = ("blue_block", "red_nest", "red_nest")

        seg_conf = dict(objects=conf_objects, sequence=conf_sequence)
        conf_dir = os.path.join(self.save_dir, "segment_conf.json")
        with open(conf_dir, 'w') as f_obj:
            json.dump(seg_conf, f_obj)

        # Convert notebook to script
        convert_cmd = "jupyter nbconvert --to script ./demo/Demonstration_Viewer.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd, check=True)

        # Run generated script
        segment_cmd = "python ./demo/Demonstration_Viewer.py {}"
        segment_cmd = segment_cmd.format(self.save_dir).split()
        subprocess.run(segment_cmd, check=True)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo/Demonstration_Viewer.py")

    def test_03_servo(self):
        control_config = dict(mode="pointcloud-abs", threshold=0.41)
        env = RobotSimEnv(task='pick_n_place', renderer='debug',
                          control='relative', max_steps=500, show_workspace=False,
                          img_size=(256, 256))

        servo_module = ServoingModule(self.save_dir,
                                      control_config=control_config,
                                      plot=False, save_dir=None)

        _, reward, _, _ = evaluate_control(env, servo_module)
        self.assertEqual(reward, 1.0)


if __name__ == '__main__':
    unittest.main()
