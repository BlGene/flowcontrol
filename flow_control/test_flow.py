import os
import json
import shutil
import unittest
import subprocess

from demo.demo_episode_recorder import start_recording_sim
from flow_control_main import evaluate_control
from gym_grasping.envs.robot_sim_env import RobotSimEnv


class TestFlowControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)

        cls.save_dir = "./tmp_test/pick_n_place"
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

    def test_01_record(self):
        start_recording_sim(self.save_dir)

    def test_02_segment(self):
        # segment the demonstration
        convert_cmd = "jupyter nbconvert --to script ./demo/Demonstration_Viewer.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd)

        conf_d = [(dict(name="color", color=(0, 0, 1), threshold=.65), dict(name="center")),
                  (dict(name="color", color=(1, 0, 0), threshold=.90), dict(name="center")),
                  (dict(name="color", color=(1, 0, 0), threshold=.90), dict(name="center"))]

        conf_dir = os.path.join(self.save_dir, "segment_conf.json")
        with open(conf_dir, 'w') as f_obj:
            json.dump(conf_d, f_obj)

        segment_cmd = "python ./demo/Demonstration_Viewer.py {} {}"
        segment_cmd = segment_cmd.format(self.save_dir, self.episode_num).split()
        subprocess.run(segment_cmd)

        # don't leave file lying around because e.g. github PEP check
        os.remove("./demo/Demonstration_Viewer.py")

    def test_03_servoing(self):

        control_config = dict(mode="pointcloud",
                              gain_xy=50,
                              gain_z=100,
                              gain_r=15,
                              threshold=0.41)
        task_name = "pick_n_place"
        robot = "kuka"
        renderer = "debug"
        control = "relative"

        env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer,
                          control=control, max_steps=500, show_workspace=False,
                          img_size=(256, 256))

        state, reward, done, info = evaluate_control(env, self.save_dir,
                                                     episode_num=self.episode_num,
                                                     control_config=control_config,
                                                     plot=False)
        self.assertEqual(reward, 1.0)


if __name__ == '__main__':
    unittest.main()
