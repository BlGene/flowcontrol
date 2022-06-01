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
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule



def split_recording(recording):
    """
    Split a recording based on waypoint names.
    """
    rec = load_rec_list(recording)

    rec_files = [rec_el.file for rec_el in rec]

    if "wp_name" in rec[0].data["info"].item():
        wp_names = [rec_el.data["info"].item()["wp_name"] for rec_el in rec]
        print("loaded waypoint names")
    else:
        wp_names = None

    wp_dones = [wp_name.endswith("_done") for wp_name in wp_names]
    for i in range(len(wp_dones)-1):
        if wp_dones[i] is True and wp_dones[i+1] is True:
            wp_dones[i] = False

    split_at = np.where(wp_dones)[0]
    split_at = [-1,] + split_at.tolist() + [len(wp_dones)]
    segments = list(zip(np.array(split_at[:-1])+1, 1+np.array(split_at[1:])))

    for i, seg in enumerate(segments):
        # check
        # print(wp_names[slice(*seg)])

        seg_save_dir = recording + f"_seg{i}"

        if os.path.isdir(seg_save_dir):
            shutil.rmtree(seg_save_dir)

        os.makedirs(seg_save_dir)
        extra_files = ["camera_info.npz", "env_info.json"]
        extra_files = [os.path.join(recording, efn) for efn in extra_files]
        seg_files = rec_files[slice(*seg)]
        for fn in seg_files + extra_files:
            fn_new = fn.replace(recording, seg_save_dir)
            shutil.copy(fn, fn_new)

        print(f"segment {i}: done copying: {len(seg_files)} files")


class ShapeSorting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)
        cls.save_dir_template = "./tmp_test/shape_sorting"
        cls.save_dir = None

    # TODO(max): sample_params False, but chaning seed still changes values.
    def test_01_record(self):
        seed = 3
        orn_options = dict(
            rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
            #rZ=R.from_euler("xyz", (0, 0, 20), degrees=True).as_quat(),
            #rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
            #rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
            #rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
            )

        for name, orn in orn_options.items():
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
            self.save_dir = save_dir
            break

    def test_02_segment(self):
        # segment the demonstration

        if self.save_dir is None:
            self.save_dir = self.save_dir_template + "_rN"

        # Convert notebook to script
        convert_cmd = "jupyter nbconvert --to script ./demo/Demonstration_Viewer.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd)

        # Run generated script
        segment_cmd = f"python ./demo/Demonstration_Viewer.py {self.save_dir}"
        subprocess.run(segment_cmd.split(), check=True)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo/Demonstration_Viewer.py")

    def test_03_servoing(self):
        if self.save_dir is None:
            self.save_dir = self.save_dir_template + "_rN"

        seed=3
        orn = R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat()
        control_config = dict(mode="pointcloud", threshold=0.41)

        env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                          initial_pose='close', max_steps=500, control='relative',
                          img_size=(256, 256),
                          sample_params=False,
                          param_info={"trapeze_pose": [[0.043, -0.60, 0.140], orn]},
                          seed=seed
                          )

        servo_module = ServoingModule(self.save_dir,
                                      episode_num=0,
                                      control_config=control_config,
                                      plot=True, save_dir=None)

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
