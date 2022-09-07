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

rec_path = './selection/tmp_test'
recs = sorted([osp.join(rec_path, file) for file in os.listdir(rec_path)])

recs_subset = recs[0:80]

rewards = np.zeros((5, len(recs_subset)))

for seed_idx, seed in enumerate(range(115, 120, 1)):
    for idx, rec in enumerate(recs_subset):

        control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
        param_info = {"task_selected": 'pick_n_place'}

        env = RobotSimEnv(task='recombination', renderer='egl', act_type='continuous',
                    initial_pose='close', max_steps=200, control='absolute-full',
                    img_size=(256, 256),
                    sample_params=False,
                    param_info=param_info,
                    seed=seed)        

        sm = ServoingModule(rec, control_config=control_config, start_paused=False, plot=False)

        _, reward, _, info = evaluate_control(env, sm, max_steps=130,initial_align=True)

        rewards[seed_idx, idx] = reward
        del env
        del sm
        
    np.savez('./selection/dataset_115_120.npz', rewards)
