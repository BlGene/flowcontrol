"""
Test servoing for the shape sorting task.
"""
import os
import time
import json
import shutil
import unittest
import subprocess
from pathlib import Path

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule

renderer = "debug"


def get_configurations(root_dir="/tmp/flow_experiments3", num_episodes=3, prefix=""):
    task="shape_sorting"
    object_selected = "trapeze"
    #object_selected = "semicircle"
    #object_selected = "oval"

    orn_options = dict(
        rR=None  # rotation is randomized
        #rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
        #rZ=R.from_euler("xyz", (0, 0, 20), degrees=True).as_quat(),
        #rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
        #rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
        #rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
        )

    os.makedirs(root_dir, exist_ok=True)

    if prefix == "":
        save_dir_template = os.path.join(root_dir, f"{task}_{object_selected}")
    else:
        save_dir_template = os.path.join(root_dir, f"{prefix}_{task}_{object_selected}")

    for seed in range(num_episodes):
        for orn_name, orn in orn_options.items():
            save_dir = save_dir_template + f"_{orn_name}"+f"_seed{seed:03d}"
            yield object_selected, orn_name, orn, seed, save_dir


def record_multi():
    demo_cfgs = get_configurations(prefix="demo")
    for object_selected, orn_name, orn, seed, save_dir in demo_cfgs:
        param_info={"object_selected": object_selected}
        env = RobotSimEnv(task='shape_sorting', renderer=renderer, act_type='continuous',
                          initial_pose='close', max_steps=200, control='absolute-full',
                          img_size=(256, 256),
                          param_randomize=("geom",),
                          param_info=param_info,
                          seed=seed)

        if os.path.isdir(save_dir):
            # lsof file if there are NSF issues.
            shutil.rmtree(save_dir)
        record_sim(env, save_dir)
        del env
        time.sleep(.5)


def segment():
    # Convert notebook to script
    convert_cmd = "jupyter nbconvert --to script ./demo/Demonstration_Viewer.ipynb"
    convert_cmd = convert_cmd.split()
    subprocess.run(convert_cmd, check=True)

    for _, _, _, seed, save_dir in get_configurations(prefix="demo"):
        segment_cmd = f"python ./demo/Demonstration_Viewer.py {save_dir}"
        subprocess.run(segment_cmd.split(), check=True)

    # Cleanup, don't leave file lying around because e.g. github PEP check
    os.remove("./demo/Demonstration_Viewer.py")


def servo():
    control_config = dict(mode="pointcloud-abs-rotz", threshold=0.40)

    demo_cfgs = get_configurations(prefix="demo")
    for _, _, _, demo_seed, demo_dir in demo_cfgs:
        run_cfgs =  get_configurations(prefix="run")
        for _, _, _, seed, save_dir in run_cfgs:
            save_dir2 = f"{save_dir}_{demo_seed:03d}"
            if Path(save_dir2).is_dir():
                shutil.rmtree(save_dir2)

            servo_module = ServoingModule(demo_dir, control_config=control_config,
                                          start_paused=False,
                                          plot=False,
                                          plot_save_dir=None)

            env = RobotSimEnv(task='shape_sorting', renderer=renderer, act_type='continuous',
                              initial_pose='close', max_steps=500, control='absolute-full',
                              img_size=(256, 256),
                              param_randomize=("geom",),
                              seed=seed)

            _, reward, _, info = evaluate_control(env, servo_module,
                                                  max_steps=130,
                                                  save_dir=save_dir2)
            print(f"Servoing completed in {info['ep_length']} steps")
            del servo_module
            del env
            time.sleep(.5)


if __name__ == '__main__':
    record_multi()
    segment()
    servo()
