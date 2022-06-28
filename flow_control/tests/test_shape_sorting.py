"""
Test servoing for the shape sorting task.
"""
import os
import shutil
import unittest
import subprocess

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


class ShapeSorting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)

        cls.object_selected = "trapeze"
        #cls.object_selected = "semicircle"
        #cls.object_selected = "oval"

        cls.orn_options = dict(
            #rR=None  # rotation is randomized
            #rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
            #rZ=R.from_euler("xyz", (0, 0, 20), degrees=True).as_quat(),
            rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
            #rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
            #rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
            )

        cls.save_dir_template = f"./tmp_test/shape_sorting_{cls.object_selected}"

    # TODO(max): sample_params False, but chaning seed still changes values.
    def test_01_record(self):
        seed = 3
        for name, orn in self.orn_options.items():
            param_info={"object_selected": self.object_selected}
            if name != "rR":
                param_info={f"{self.object_selected}_pose": [[0.043, -0.60, 0.140], orn]}

            env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=200, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info=param_info,
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
        subprocess.run(convert_cmd, check=True)

        for name in self.orn_options:
            # Run generated script
            save_dir = self.save_dir_template + f"_{name}"
            segment_cmd = f"python ./demo/Demonstration_Viewer.py {save_dir}"
            subprocess.run(segment_cmd.split(), check=True)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo/Demonstration_Viewer.py")

    def test_03_servo(self):
        seed = 3
        control_config = dict(mode="pointcloud-abs", threshold=0.40)

        for name, orn in self.orn_options.items():
            param_info={}
            if name != "rR":
                param_info={f"{self.object_selected}_pose": [[0.0, -0.60, 0.140], orn]}

            save_dir = self.save_dir_template + f"_{name}"

            servo_module = ServoingModule(save_dir,
                                          control_config=control_config,
                                          plot=True, save_dir=None,
                                          start_paused=False)

            env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=500, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info=param_info,
                              seed=seed)

            _, reward, _, info = evaluate_control(env, servo_module)
            print(f"Servoing completed in {info['ep_length']} steps")
            self.assertEqual(reward, 1.0)


    # def test_05_split(self):
    #    name = "rZ"
    #    save_dir = self.save_dir + f"_{name}"
    #    split_recording(save_dir)


if __name__ == '__main__':
    unittest.main()
