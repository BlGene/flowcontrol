import os
import json
import time
import subprocess

import cv2
import ipdb
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule
from demo_selection import *

def get_errors(error_file):
    fp = open(error_file)
    errors = json.load(fp)

    return errors

def get_best_tasks(errors, seed):
    if str(seed) in errors.keys():
        seed_errors = errors[str(seed)]
    else:
        print("Bad seed value")
        return

    best_idx = []

    for i in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
        # ipdb.set_trace()
        subset = seed_errors[0:i]
        best_idx.append(np.argmin(subset))
    return best_idx

def plot_rewards(reward_file, savefig, title):
    fp = open(reward_file)
    rewards = json.load(fp)

    total_rew = None

    for seed, val in rewards.items():
        if total_rew is None:
            total_rew = val
        else:
            total_rew = [x + y for x, y in zip(total_rew, val)]

    keys = list(rewards.keys())

    length = len(rewards[keys[0]])
    x = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

    total_rew = [rew / len(keys) for rew in total_rew]

    plt.plot()
    
    plt.plot(x, total_rew, ".-")
    plt.xlabel("#Recordings")
    plt.ylabel("Mean Rewards")
    plt.title(f"Mean rewards vs #Recordings used for {title}")
    plt.savefig(savefig)

def plot_counter(counter_file, reward_file, savefig):
    fp = open(counter_file)
    counter_data = json.load(fp)

    fp = open(reward_file)
    reward_data = json.load(fp)

    keys = list(reward_data.keys())

    reward_1 = []
    reward_0 = []

    for key, val in counter_data.items():
        reward_1 += [val for idx, val in enumerate(counter_data[key]) if reward_data[key][idx] == 1.0]
        reward_0 += [val for idx, val in enumerate(counter_data[key]) if reward_data[key][idx] == 0.0]

    print(reward_1, reward_0)
    plt.hist(reward_0, bins=10, alpha=0.5, label="Reward=0")
    plt.hist(reward_1, bins=10, alpha=0.5, label="Reward=1")
    plt.legend(loc='upper right')
    plt.ylabel("Count")
    plt.xlabel("#Steps")
    plt.savefig(savefig)


if __name__ == '__main__':
    s_time = time.process_time()
    exp = "pnp_dummy"

    result_path = f"./conditional/rewards_{exp}"
    os.makedirs(result_path, exist_ok=True)
    reward_file = os.path.join(result_path, f'reward_{exp}.json')
    counter_file = os.path.join(result_path, f'counter_{exp}.json')
    error_file = "./conditional/errors_pnp/errors_all.json"

    rec_paths = ['./pnp_tmp_test']

    all_recordings = []
    for path in rec_paths:
        files = os.listdir(path)
        all_recordings += [os.path.join(path, file) for file in files]

    recordings = [rec for rec in sorted(all_recordings)]

    # Use first 75 recordings
    recordings = sorted(recordings)[0:75]
    control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)
    param_info = {"task_selected": 'pick_n_place'}

    test_seeds = range(100, 120, 1)

    # store_errors(recordings, test_seeds, error_file)
    errors = get_errors(error_file)

    rewards = {}
    counter = {}
    times = []

    for seed in test_seeds:
        seed_dir = os.path.join(result_path, f"{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        rewards[seed] = []
        counter[seed] = []

        best_idx = get_best_tasks(errors, seed)
        print(best_idx)
        already_run = []
        last_rew = 0
        last_ep_len = 0

        print("--------------------------------------------------------")

        for idx in best_idx:
            rec = recordings[idx]

            if idx in already_run:
                print("Same recording as before")
                rewards[seed].append(last_rew)
                counter[seed].append(last_ep_len)
                continue

            # print("--------------------------------------------------------")
            print(f"Seed: {seed}, Using Recording: {rec}")
            # print("--------------------------------------------------------")

            continue

            start_time = time.process_time()
            env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                              initial_pose='close', max_steps=500, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info=param_info,
                              seed=seed)

            servo_module = ServoingModule(rec, control_config=control_config,
                                          plot=False)

            _, reward, _, info = evaluate_control(env, servo_module,
                                                  max_steps=130, save_dir=seed_dir)
            print(f"Servoing completed in {info['ep_length']} steps")

            end_time = time.process_time()

            times.append((end_time - start_time))

            rewards[seed].append(reward)
            counter[seed].append(info['ep_length'])

            last_rew = reward
            last_ep_len = info["ep_length"]
            already_run.append(idx)

            del servo_module
            del env

            subproc_cmd = f'ffmpeg -framerate 8 -i {seed_dir}/frame_%06d.jpg -r 25 -pix_fmt yuv420p ' \
                          f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {seed_dir}/video_{rec[-6:]}.mp4'

            # Run subprocess using the command
            subprocess.run(subproc_cmd, check=True, shell=True)

            for item in os.listdir(seed_dir):
                if item.endswith('.jpg'):
                    os.remove(os.path.join(seed_dir, item))

    print(times, np.mean(times))

    with open(reward_file, 'w') as outfile:
        json.dump(rewards, outfile)

    with open(counter_file, 'w') as outfile:
        json.dump(counter, outfile)

    plot_rewards(reward_file, os.path.join(result_path, 'reward_plot.png'), exp)
    plt.cla()

    plot_counter(counter_file, reward_file, os.path.join(result_path, 'step_counter.png'))

    e_time = time.process_time()
    print(f"This took {e_time - s_time} seconds")






