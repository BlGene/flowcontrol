"""
Test servoing for the shape sorting task.
"""
import os
import shutil
import unittest
import subprocess
from math import pi

import ipdb
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.record_sim import record_sim
from flow_control.servoing.runner import evaluate_control
from flow_control.servoing.module import ServoingModule

from cloning.networks.behavioral_cloning import *
from cloning.data_utils.data import *
from torchvision import transforms as T
import numpy as np

class ShapeSorting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("./tmp_test", exist_ok=True)

        cls.object_selected = "trapeze"
        #cls.object_selected = "semicircle"
        #cls.object_selected = "oval"

        cls.orn_options = dict(
            rR=None,  # rotation is randomized
            #rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
            #rZ=R.from_euler("xyz", (0, 0, 90), degrees=True).as_quat(),
            #rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
            #rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
            #rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
            )

        cls.save_dir_template = f"./tmp_test/shape_sorting_{cls.object_selected}"
        cls.save_dir_servo = f"./tmp_test/run_shape_sorting_{cls.object_selected}"

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
                              task_info=dict(object_rot_range=pi/2),
                              param_randomize=("geom",),
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
        convert_cmd = "jupyter nbconvert --to script ./demo_segment/Segment_Color.ipynb"
        convert_cmd = convert_cmd.split()
        subprocess.run(convert_cmd, check=True)

        for name in self.orn_options:
            # Run generated script
            save_dir = self.save_dir_template + f"_{name}"
            segment_cmd = f"python ./demo_segment/Segment_Color.py {save_dir}"
            subprocess.run(segment_cmd.split(), check=True)

        # Cleanup, don't leave file lying around because e.g. github PEP check
        os.remove("./demo_segment/Segment_Color.py")

    def test_03_servo(self):
        seed = 10
        control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25, threshold_far=1.5)

        for name, orn in self.orn_options.items():
            param_info={}
            if name != "rR":
                param_info={f"{self.object_selected}_pose": [[0.0, -0.60, 0.140], orn]}

            save_dir = self.save_dir_template + f"_{name}"

            servo_module = ServoingModule(save_dir, control_config=control_config, start_paused=False, plot=True,
                                          plot_save_dir=None, flow_module='RAFT')

            env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=500, control='absolute-full',
                              img_size=(256, 256),
                              task_info=dict(object_rot_range=pi/2),
                              param_randomize=("geom",),
                              param_info=param_info,
                              seed=seed)

            _, reward, _, info = evaluate_control(env, servo_module)
            print(f"Servoing completed in {info['ep_length']} steps")
            self.assertEqual(reward, 1.0)

    def test_04_servo(self):
        seed = 40
        control_config = dict(mode="pointcloud-abs-rotz", threshold=0.40)

        model = nn.DataParallel(Cloning()).cuda()
        model.eval()

        load_model = "/home/argusm/lmb_abhi/behavioral_cloning/ckpts_first_train/model_35.pth"
        model.load_state_dict(torch.load(load_model))

        transform = T.Compose([T.ToTensor()])

        dataset = DemoDataset(data_dir="/home/argusm/lang/flowcontrol/flow_control/demo_selection/temp/")

        for name, orn in self.orn_options.items():
            param_info = {}
            if name != "rR":
                param_info={f"{self.object_selected}_pose": [[0.0, -0.60, 0.140], orn]}

            save_dir = self.save_dir_template + f"_{name}"

            env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=500, control='absolute-full',
                              img_size=(256, 256),
                              task_info=dict(object_rot_range=pi/2),
                              param_randomize=("geom",),
                              param_info=param_info,
                              seed=seed)

            for idx in range(0, 100000):
                im, trans, rot, gripper = dataset.__getitem__(idx)
                # gripper = np.atleast_1d(gripper)
                state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                # state, _, _, _ = env.step(None)
                rgb_img = state['rgb_gripper']

                rgb_img = transform(rgb_img)
                rgb_img = rgb_img.unsqueeze(0).cuda()

                # trans, rot, gripper = model(rgb_img)
                #
                # trans = trans.detach().cpu().numpy()
                # rot = rot.detach().cpu().numpy()
                # gripper = gripper.detach().cpu().numpy()

                gripper = np.round(gripper)

                print(trans, rot, gripper)
                # ipdb.set_trace()

                # cur_pos = state['robot_state']['tcp_pos']
                # cur_orn = state['robot_state']['tcp_orn']
                # cur_abs = pos_orn_to_matrix(cur_pos, cur_orn)
                cur_abs = env.robot.get_tcp_pose()

                rel = pos_orn_to_matrix(trans, rot)

                trf = cur_abs @ rel
                abs_new = [list(x) for x in matrix_to_pos_orn(trf)]

                # ipdb.set_trace()
                action = dict(motion=(abs_new[0], abs_new[1], gripper), ref="abs", blocking=True)

                state, reward, done, info = env.step(action)

                if reward == 1.0:
                    print("Ended with reward 1")
                    break


    # def test_05_split(self):
    #    name = "rZ"
    #    save_dir = self.save_dir + f"_{name}"
    #    split_recording(save_dir)


if __name__ == '__main__':
    unittest.main()
