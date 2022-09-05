"""
Test servoing for the shape sorting task.
"""
import os
import shutil
import unittest
import subprocess
import numpy as np
import ipdb
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


class Recombination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.object_selected = "trapeze"
        cls.task_selected = "shape_sorting"
        cls.save_dir_template = f"./tmp_test/{cls.task_selected}_{cls.object_selected}"

    def test_01_record(self):
        n_digits = 6
        for seed in range(80):
            param_info = {"object_selected": self.object_selected,
                          "task_selected": self.task_selected}

            env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=200, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info=param_info,
                              seed=seed)

            save_dir = self.save_dir_template + "_rR_" + f"{seed:0{n_digits}d}"
            if os.path.isdir(save_dir):
                # lsof file if there are NSF issues.
                shutil.rmtree(save_dir)
            reward = record_sim(env, save_dir)
            if reward != 1.0:
                shutil.rmtree(save_dir)

            del env

    def test_02_segment(self):
        # segment the demonstration

        # Convert notebook to script
        convert_cmd = "jupyter nbconvert --to script ./demo/Demonstration_Viewer.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd, check=True)

        folder = "./tmp_test"

        for rec in sorted(os.listdir(folder)):
            save_dir = os.path.join(folder, rec)

            try:
                # Run generated script
                segment_cmd = f"python ./demo/Demonstration_Viewer.py {save_dir}"
                subprocess.run(segment_cmd.split(), check=True)
            except:
                shutil.rmtree(save_dir)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo/Demonstration_Viewer.py")

    def test_03_servo(self):
        seed = 110
        control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
        param_info = {"task_selected": 'pick_n_place'}

        recs = ['./recombination/tmp_test_new_split1/ss/shape_sorting_trapeze_rR_000055_seg0',
                './recombination/tmp_test_new_split1/ss/pick_n_place_trapeze_rR_000089_seg1']

        env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                          initial_pose='close', max_steps=500, control='absolute-full',
                          img_size=(256, 256),
                          sample_params=False,
                          param_info=param_info,
                          seed=seed)

        reward = 0.0

        for idx, rec in enumerate(recs):
            # ipdb.set_trace()
            initial_align = True
            if idx > 0:
                initial_align = False
            servo_module = ServoingModule(rec, control_config=control_config, start_paused=False, plot=False)
            _, reward, _, info = evaluate_control(env, servo_module, max_steps=130, save_dir=None,
                                                  initial_align=initial_align)
            print(f"Servoing completed in {info['ep_length']} steps")
            if reward == 1.0:
                print(f'Reward: {reward}')
                break

        self.assertEqual(reward, 1.0)

    def test_03_servo_all(self):
        # bidx = {107: 10, 106: 54, 117: 11, 108: 44, 101: 27, 100: 27,
        # 102: 59, 113: 52, 105: 52, 110: 43, 104: 53, 118: 9, 103: 61, 112: 33,
        # 109: 35, 111: 49, 114: 66, 119: 33, 115: 17, 116: 9}
        bidx = {100: 43, 101: 38, 102: 56, 103: 38, 104: 66, 105: 70, 106: 50, 107: 38, 108: 13, 109: 38, 110: 12,
                111: 6, 112: 62, 113: 54, 114: 36, 115: 38, 116: 70, 117: 38, 118: 63, 119: 38}

        seeds = range(100, 120, 1)
        rec_path = './recombination/tmp_test_new_split1/ss'
        recs = sorted([os.path.join(rec_path, f) for f in os.listdir(rec_path)])
        rewards = []

        for seed in seeds:
            rec = recs[2 * bidx[seed]]
            print(rec)
            # continue
            control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
            param_info = {"task_selected": 'pick_n_place'}

            seed_recs = [rec, './recombination/tmp_test_new_split1/ss/pick_n_place_trapeze_rR_000089_seg1']

            env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=500, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info=param_info,
                              seed=seed)

            reward = 0.0

            for idx, r in enumerate(seed_recs):
                initial_align = True
                if idx > 0:
                    initial_align = False
                servo_module = ServoingModule(r, control_config=control_config, start_paused=False, plot=False)
                _, reward, _, info = evaluate_control(env, servo_module, max_steps=130, save_dir="",
                                                    initial_align=initial_align)
                print(f"Servoing completed in {info['ep_length']} steps")
                if reward == 1.0:
                    print(f'Reward: {reward}')
                    break
            rewards.append(reward)
            del env
            del servo_module

        print(rewards, np.mean(rewards))
        # self.assertEqual(reward, 1.0)

    def test_04_pnp(self):
        pnp_rec_path = "./recombination/tmp_test/pick_n_place"
        rec_files = sorted([os.path.join(pnp_rec_path, file) for file in os.listdir(pnp_rec_path)])
        seeds = range(100, 120, 1)
        rewards = np.zeros((len(rec_files), len(seeds)))

        control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
        param_info = {"task_selected": 'pick_n_place'}

        for rec_idx, rec in enumerate(rec_files):
            for seed_idx, seed in enumerate(seeds):
                env = RobotSimEnv(task='recombination', renderer='egl', act_type='continuous',
                                        initial_pose='close', max_steps=500, control='absolute-full',
                                        img_size=(256, 256),
                                        sample_params=False,
                                        param_info=param_info,
                                        seed=seed)
                sm = ServoingModule(rec, control_config=control_config, start_paused=False, plot=False)
                _, reward, _, info = evaluate_control(env, sm, max_steps=130, initial_align=True)

                rewards[rec_idx, seed_idx] = reward

                np.savez('pnp_single_rewards.npz', rewards)
    def test_05_errors(self):
        # Convert notebook to script
        convert_cmd = "jupyter nbconvert --to script ./demo/recombination_efficient.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd, check=True)

        rec_path = "./tmp_test_split"

        # Run generated script
        segment_cmd = f"python ./demo/recombination_efficient.py {rec_path}"
        subprocess.run(segment_cmd.split(), check=True)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo/recombination_efficient.py")

if __name__ == '__main__':
    unittest.main()
