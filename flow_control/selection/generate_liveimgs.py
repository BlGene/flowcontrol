import os
import os.path as osp
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
import cv2

rec_path = './recombination/tmp_test_new_ss'
recs = sorted([osp.join(rec_path, file) for file in os.listdir(rec_path)])

recs_subset = recs[0:80]
print(len(recs_subset))

live_dir = './selection/debug_live_imgs_test'
os.makedirs(live_dir, exist_ok=True)

demo_dir = './selection/debug_demo_dir'
os.makedirs(demo_dir, exist_ok=True)

n_digits = 6

for seed_idx, seed in enumerate(range(100, 120, 1)):

    control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
    param_info = {"task_selected": 'pick_n_place'}

    env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                initial_pose='close', max_steps=200, control='absolute-full',
                img_size=(256, 256),
                sample_params=False,
                param_info=param_info,
                seed=seed)

    state, _, _, _ = env.step(None)
    live_img = state['rgb_gripper']
    live_img_path = osp.join(live_dir, f"{seed:0{n_digits}d}.jpg")
    cv2.imwrite(live_img_path, cv2.cvtColor(live_img, cv2.COLOR_RGB2BGR))

    del env

for idx, rec in enumerate(recs_subset):
    print(idx)
    img_path = osp.join(rec, 'frame_000000.jpg')
    demo_path = osp.join(demo_dir, f"{idx:0{n_digits}d}.jpg")
    shutil.copyfile(img_path, demo_path)
