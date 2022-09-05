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


def select_demo(servo_modules, live_rgb, best_tasks, goal_frame, use_goal=True):
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

    # Multipliers for Front and Rear errors_old
    alpha, beta = 1.0, 1.0

    for t, s in servo_modules:
        error_front, error_rear = 0.0, 0.0
        if use_goal:
            last_rec_im = s.demo.steps[-5].cam.get_image()[0]
            flow_rear = s.flow_module.step(goal_frame, last_rec_im)
            warped_rear = s.flow_module.warp_image(goal_frame / 255.0, flow_rear)

            demo_mask_rear = s.demo.fg_masks[-5]
            mask_rear = np.zeros((256, 256))
            mask_rear[demo_mask_rear == True] = 255.0
            error_rear = ((warped_rear - (last_rec_im / 255.0))
                          ** 2.0).sum(axis=2) * mask_rear

            if mask_rear.sum() == 0.0:
                error_rear = 2.0
            else:
                error_rear = error_rear.sum() / mask_rear.sum()

        first_rec_im = s.demo.steps[0].cam.get_image()[0]
        flow_front = s.flow_module.step(live_rgb, first_rec_im)

        warped_front = s.flow_module.warp_image(live_rgb / 255.0, flow_front)
        # diff = np.clip(abs(demo_rgb - live_rgb), 0.0, 255.0)
        # mean = np.mean((demo_rgb, live_rgb), axis=0)
        # ipdb.set_trace()

        # cv2.imshow("warped", warped)
        # cv2.imshow("Demo", demo_rgb)
        # cv2.imshow("Live", live_rgb)
        # cv2.imshow("mean", mean/255.0)
        # cv2.waitKey(0)

        # Logical demo mask
        demo_mask_front = s.demo.fg_masks[0]
        mask_front = np.zeros((256, 256))
        mask_front[demo_mask_front == True] = 255.0

        error_front = ((warped_front - (first_rec_im / 255.0))
                       ** 2.0).sum(axis=2) * mask_front

        if mask_front.sum() == 0.0:
            error_front = 2.0
        else:
            error_front = error_front.sum() / mask_front.sum()

        error = error_front * alpha + error_rear * beta
        print(f"Error: {error}")

        if error < best_error and t not in best_tasks:
            best_error = error
            best_task = t
            best_servo_module = s

    return best_servo_module, best_task

rewards_list = []

rec_path = './tmp_test_segmented/tmp_test'
result_path = './recombination/results'
os.makedirs(result_path, exist_ok=True)

recordings = [os.path.join(rec_path, rec) for rec in os.listdir(rec_path)]
control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
# Set up servo modules
servo_modules = [(t, ServoingModule(t, control_config=control_config,
                                    plot=False, save_dir=None)) for t in sorted(recordings)]

seeds = [0]
goal_im_path = './demo/frame_000012.jpg'
goal_frame = cv2.imread(goal_im_path)

for seed in range(100, 120, 1):
    seed_dir = os.path.join(result_path, f"{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    seed_rewards = []
    best_tasks = []

    # Instantiate environment
    env = RobotSimEnv(task='pick_n_place', renderer='debug', act_type='continuous',
                      initial_pose='close', max_steps=500, control='absolute-full',
                      img_size=(256, 256),
                      sample_params=False,
                      seed=seed)

    reward = 0.0

    for trial_num in range(3):
        # Get live state
        state, _, _, _ = env.step(None)
        live_rgb = state['rgb_gripper']

        _, best_task = select_demo(servo_modules, live_rgb, best_tasks, goal_frame, use_goal=bool(trial_num))
        print(f"Best Task selected: {best_task}")
        best_tasks.append(best_task)

        best_servo_module = ServoingModule(best_task, control_config=control_config,
                                        plot=False, save_dir=f"./conditional/results/{seed}")
        initial_align = True
        if trial_num > 0:
            initial_align = False
        _, reward, _, info = evaluate_control(env, best_servo_module, max_steps=100,
                                              save_dir=seed_dir, initial_align=initial_align)

        subproc_cmd = f'ffmpeg -framerate 8 -i {seed_dir}/frame_%06d.jpg -r 25 -pix_fmt yuv420p ' \
                      f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {seed_dir}/video{trial_num}.mp4'

        # Run subprocess using the command
        subprocess.run(subproc_cmd, check=True, shell=True)

        for file in os.listdir(seed_dir):
            if file.endswith('.jpg'):
                os.remove(os.path.join(seed_dir, file))

        if reward == 1.0:
            # We can exit now
            print(f"Servoing completed in {info['ep_length']} steps")
            print(f"Reward: {reward}")
            seed_rewards.append(reward)
            break

        seed_rewards.append(reward)

        del best_servo_module

    rewards_list.append(seed_rewards)



    del env

print(rewards_list)
print(rewards_list, np.mean([rew[-1] for rew in rewards_list]))
print(best_tasks)


