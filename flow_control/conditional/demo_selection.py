import os
import json
import time

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule

def demo_selection_errors(servo_modules, live_rgb):
    """
    Selects the demonstration with the minimum reprojection error

    Args:
        servo_modules:
        live_rgb: Array with the live view

    Returns:
        best_servo_module:
    """

    demo_errors = []

    for t, s in servo_modules:
        print(f"Task: {t}")
        demo_rgb = s.demo.cam.get_image()[0]
        flow = s.flow_module.step(live_rgb, demo_rgb)
        warped = s.flow_module.warp_image(live_rgb / 255.0, flow)

        # Logical demo mask
        demo_mask = s.demo.fg_masks[0]

        mask = np.zeros((256, 256))
        mask[demo_mask == True] = 255.0

        error = ((warped - (demo_rgb / 255.0)) ** 2.0).sum(axis=2) * mask  # s.demo.mask
        error = error.sum() / mask.sum()
        demo_errors.append(error)

    return demo_errors

def store_errors(recordings, seeds, error_file):
    errors = {}
    control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)

    servo_modules = [(t, ServoingModule(t, control_config=control_config,
                                        plot=False, save_dir=None)) for t in recordings]
    for seed in seeds:
        env = RobotSimEnv(task='shape_sorting', renderer='egl', act_type='continuous',
                          initial_pose='close', max_steps=500, control='absolute-full',
                          img_size=(256, 256),
                          sample_params=False,
                          seed=seed)

        # Get live state
        state, _, _, _ = env.step(None)
        live_rgb = state['rgb_gripper']

        errors[seed] = demo_selection_errors(servo_modules, live_rgb)

        del env

    with open(error_file, 'w') as outfile:
        json.dump(errors, outfile)

def compare_errors(error_file1, error_file2):
    fp = open(error_file1)
    errors_1 = json.load(fp)

    fp = open(error_file2)
    errors_2 = json.load(fp)

    for keys, values in errors_1.items():
        e1 = values
        e2 = errors_2[keys]
        x = list(np.arange(len(e1)))
        # plt.scatter(x, e1, color='b')
        plt.scatter(e1, e2, color='r')
        plt.show()

if __name__ == '__main__':
    error_file = "errors_all.json"
    seeds = range(100, 120, 1)
    #
    # rec_path = './tmp_test'
    #
    # store_errors(rec_path, seeds, error_file)

    rec_paths = ['./tmp_test', './tmp_test_oval', './tmp_test_semicircle']

    all_recordings = []
    for path in rec_paths:
        files = os.listdir(path)
        all_recordings += [os.path.join(path, file) for file in files]

    recordings = [rec for rec in sorted(all_recordings)]

    store_errors(recordings, seeds, error_file)

    # error_file1 = 'errors_new.json'
    # error_file2 = 'errors_new_25.json'
    #
    # compare_errors(error_file1, error_file2)






