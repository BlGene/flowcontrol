import os
import shutil
import unittest
import subprocess

import cv2
import ipdb
import numpy as np

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule

orn_options = dict(
    rR=None,
    # rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
    # rZ=R.from_euler("xyz", (0, 0, 20), degrees=True).as_quat(),
    # rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
    # rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
    # rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
)


def select_demo(servo_modules, live_rgb):
    """
    Selects the demonstration with the minimum reprojection error

    Args:
        servo_modules:
        live_rgb: Array with the live view

    Returns:
        best_servo_module:
    """

    best_task = None
    best_servo_module = None
    best_error = np.inf

    idx = 0

    for t, s in servo_modules:
        print(f"Task: {t}")
        demo_rgb = s.demo.cam.get_image()[0]
        flow = s.flow_module.step(live_rgb, demo_rgb)
        warped = s.flow_module.warp_image(live_rgb / 255.0, flow)
        # diff = np.clip(abs(demo_rgb - live_rgb), 0.0, 255.0)
        mean = np.mean((demo_rgb, live_rgb), axis=0)
        # ipdb.set_trace()

        result_path = os.path.join("./presentation_plots", str(idx))
        os.makedirs(result_path, exist_ok=True)

        cv2.imwrite(os.path.join(result_path, "warped.jpg"), warped)
        cv2.imwrite(os.path.join(result_path, "demo.jpg"), demo_rgb)
        cv2.imwrite(os.path.join(result_path, "live.jpg"), live_rgb)
        cv2.imwrite(os.path.join(result_path, "mean.jpg"), mean/255.0)
        # cv2.waitKey(0)

        # Logical demo mask
        demo_mask = s.demo.fg_masks[0]

        mask = np.zeros((256, 256))
        mask[demo_mask == True] = 255.0

        cv2.imwrite(os.path.join(result_path, "mask.jpg"), mask)
        # cv2.waitKey(0)
        idx += 1

        error = ((warped - (demo_rgb / 255.0)) ** 2.0).sum(axis=2) * mask  # s.demo.mask
        error = error.sum() / mask.sum()
        print(f"Error: {error}")

        if error < best_error:
            best_error = error
            best_task = t
            best_servo_module = s

    return best_servo_module, best_task


rewards = []
best_tasks = []
rec_path = './recordings/tmp_test_old'
result_path = './conditional/results_tmp'
os.makedirs(result_path, exist_ok=True)

recordings = [os.path.join(rec_path, rec) for rec in os.listdir(rec_path)]
control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
# Set up servo modules
servo_modules = [(t, ServoingModule(t, control_config=control_config,
                                    plot=False, save_dir=None)) for t in recordings]
seeds = [15, 27, 130, 145, 250, 362, 278, 485, 596, 473,
         775, 38, 2, 56, 65, 87, 178, 4539, 9729, 111]
new_seeds = [95, 37, 100, 150, 205, 323, 842, 145, 956, 743, 777,
             238, 424, 564, 654, 876, 923, 453, 972, 1024]
for seed in new_seeds:
    seed_dir = os.path.join(result_path, f"{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Instantiate environment
    env = RobotSimEnv(task='shape_sorting', renderer='egl', act_type='continuous',
                      initial_pose='close', max_steps=500, control='absolute-full',
                      img_size=(256, 256),
                      sample_params=False,
                      seed=seed)

    # Get live state
    state, _, _, _ = env.step(None)
    live_rgb = state['rgb_gripper']

    _, best_task = select_demo(servo_modules, live_rgb)
    print(f"Best Task selected: {best_task}")
    best_tasks.append(best_task)

    best_servo_module = ServoingModule(best_task, control_config=control_config,
                                    plot=False, save_dir=f"./conditional/results/{seed}")

    # _, reward, _, info = evaluate_control(env, best_servo_module, max_steps=500)
    # print(f"Servoing completed in {info['ep_length']} steps")

    # rewards.append(reward)

    # subproc_cmd = f'ffmpeg -framerate 8 -i {seed_dir}/frame_%06d.jpg -r 25 -pix_fmt yuv420p ' \
    #               f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {seed_dir}/video.mp4'
    #
    # # Run subprocess using the command
    # subprocess.run(subproc_cmd, check=True, shell=True)

    del env

print(rewards, np.mean(rewards))
print(best_tasks)


