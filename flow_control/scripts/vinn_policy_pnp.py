"""
Test servoing for the shape sorting task.
"""
import os
import shutil
import unittest
import subprocess
from math import pi
import json

import ipdb
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.record_sim import record_sim
from flow_control.servoing.runner import evaluate_control
from flow_control.servoing.module import ServoingModule

from cloning.networks.behavioral_cloning import *
from cloning.data_utils.data import *
from torchvision import transforms as T
import numpy as np


def vinn_policy():
    seed = 3
    control_config = dict(mode="pointcloud-abs-rotz", threshold=0.40)

    task_variant = "rP"
    object_selected = 'trapeze'
    task = 'pick_n_place'

    param_info = {"object_selected": object_selected, "task_selected": task}

    env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                      initial_pose='close', max_steps=2000, control='absolute-full',
                      img_size=(256, 256),
                      param_randomize=("geom",),
                      param_info=param_info,
                      task_info=dict(object_rot_range={"rP": pi/2., "rR": pi/6.}[task_variant]),
                      seed=seed)

    if task_variant == "rP":
        assert env.params.variables[f"{object_selected}_pose"]["d"][3] == pi/2.
    elif task_variant == "rR":
        assert env.params.variables[f"{object_selected}_pose"]["d"][3] == pi/6.

    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                       std=torch.tensor([0.229, 0.224, 0.225]))])

    vinn_model = models.resnet50(pretrained=False)
    load_model = '/home/argusm/lmb_abhi/VINN/vinn_byol_sim/BYOL_95__pretrained_0.pt'

    state_dict = torch.load(load_model)

    vinn_model.load_state_dict(state_dict['model_state_dict'])

    arch = list(vinn_model.children())

    # del arch[-2]  # AvgPool
    del arch[-1]  # FC

    vinn_model = nn.Sequential(*arch)

    vinn_model.cuda()
    vinn_model.eval()

    for module in vinn_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    features = np.load('/home/argusm/lmb_abhi/VINN/dataloaders/features.npz')['arr_0']

    with open('/home/argusm/lmb_abhi/VINN/dataloaders/actions.json', 'r') as f_obj:
        actions = json.load(f_obj)

    action = None
    max_iterations = 2000

    for idx in range(max_iterations):
        state, reward, done, info = env.step(action)

        if reward == 1.0:
            print("Ended with reward 1")
            break

        rgb = state['rgb_gripper']
        rgb = transform(rgb)

        rgb = rgb.unsqueeze(0).cuda()

        out = vinn_model(rgb)
        out = out.squeeze(2).squeeze(2).detach().cpu().numpy()

        feature_diff = np.linalg.norm(features - out, axis=1)

        k_nn = np.argsort(feature_diff)[0:20]
        print(k_nn)

        positions, orientations, gripper_actions, weights = [], [], [], []
        for neighbor_idx in k_nn:
            pos, orn = actions[str(neighbor_idx)]['pos'], actions[str(neighbor_idx)]['orn']
            gripper_action = int(actions[str(neighbor_idx)]['gripper'])

            weight = np.exp(-feature_diff[neighbor_idx])
            weights.append(weight)

            positions.append(np.array(pos) * weight)
            orientations.append(np.array(orn) * weight)

            gripper_actions.append(np.array(gripper_action))

        positions = np.stack(positions).squeeze()
        orientations = np.stack(orientations).squeeze()
        gripper_actions = np.stack(gripper_actions)

        # import ipdb
        # ipdb.set_trace()

        best_pos, best_orn = np.sum(positions, axis=0) / np.sum(weights), np.sum(orientations, axis=0) / np.sum(weights)
        best_pos, best_orn = positions[10, :] * 100, orientations[10, :] * 5

        unique_elements, counts = np.unique(gripper_actions, return_counts=True)
        best_gripper_action = int(unique_elements[np.argmax(counts)])

        # Setting action for next time step
        action = dict(motion=(best_pos, best_orn, best_gripper_action), ref="rel", blocking=True)

    #
    # def test_04_servo(self):
    #     seed = 0
    #     control_config = dict(mode="pointcloud-abs-rotz", threshold=0.40)
    #
    #     model = nn.DataParallel(Cloning()).cuda()
    #     model.eval()
    #
    #     load_model = "/home/argusm/lmb_abhi/behavioral_cloning/ckpts_first_train/model_35.pth"
    #     model.load_state_dict(torch.load(load_model))
    #
    #     transform = T.Compose([T.ToTensor()])
    #
    #     dataset = DemoDataset(data_dir="/tmp/temp")
    #
    #     for name, orn in self.orn_options.items():
    #         param_info = {}
    #         if name != "rR":
    #             param_info={f"{self.object_selected}_pose": [[0.0, -0.60, 0.140], orn]}
    #
    #         save_dir = self.save_dir_template + f"_{name}"
    #
    #         env = RobotSimEnv(task='shape_sorting', renderer='debug', act_type='continuous',
    #                           initial_pose='close', max_steps=500, control='absolute-full',
    #                           img_size=(256, 256),
    #                           task_info=dict(object_rot_range=pi/2),
    #                           param_randomize=("geom",),
    #                           param_info=param_info,
    #                           seed=seed)
    #
    #         for idx in range(0, 100000):
    #             # im, trans, rot, gripper = dataset.__getitem__(idx)
    #             # gripper = np.atleast_1d(gripper)
    #             state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             # state, _, _, _ = env.step(None)
    #             rgb_img = state['rgb_gripper']
    #
    #             rgb_img = transform(rgb_img)
    #             rgb_img = rgb_img.unsqueeze(0).cuda()
    #
    #             trans, rot, gripper = model(rgb_img)
    #             #
    #             trans = trans.detach().cpu().numpy()
    #             rot = rot.detach().cpu().numpy()
    #             gripper = gripper.detach().cpu().numpy()
    #
    #             gripper = np.round(gripper)
    #
    #             print(trans, rot, gripper)
    #             # ipdb.set_trace()
    #
    #             # cur_pos = state['robot_state']['tcp_pos']
    #             # cur_orn = state['robot_state']['tcp_orn']
    #             # cur_abs = pos_orn_to_matrix(cur_pos, cur_orn)
    #             cur_abs = env.robot.get_tcp_pose()
    #
    #             rel = pos_orn_to_matrix(trans[0], rot[0])
    #
    #             trf = cur_abs @ rel
    #             abs_new = [list(x) for x in matrix_to_pos_orn(trf)]
    #
    #             # ipdb.set_trace()
    #             action = dict(motion=(abs_new[0], abs_new[1], gripper[0]), ref="abs", blocking=True)
    #
    #             state, reward, done, info = env.step(action)
    #
    #             if reward == 1.0:
    #                 print("Ended with reward 1")
    #                 break


    # def test_05_split(self):
    #    name = "rZ"
    #    save_dir = self.save_dir + f"_{name}"
    #    split_recording(save_dir)


if __name__ == '__main__':
    vinn_policy()
