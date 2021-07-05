"""
Testing file for development, to experiment with environments.
"""
import math
import logging
import platform

import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule

from pdb import set_trace
inv = np.linalg.inv


def save_frame(name, state, reward, done, info):
    kwds = dict(state=state, reward=reward, done=done)
    for k, v in info.items():
        kwds["__info__"+k] = v
    np.savez_compressed(name, **kwds)


def load_frame(name):
    load_dict = np.load(name)
    info = {}
    for key, value in load_dict.items():
        if key.startswith("__info__"):
            info[key.replace("__info__", "")] = value
    state = load_dict["state"]
    reward = float(load_dict["reward"])
    done = int(load_dict["done"])
    return state, reward, done, info


def tcp_pose_from_info(info):
    curr_pos = info["tcp_pose"]
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler('xyz', curr_pos[3:6]).as_matrix()
    matrix[:3, 3] = curr_pos[:3]
    return matrix


def evaluate_control(env, servo_module, start_paused=False,  max_steps=1000):
    """
    Function that runs the policy.
    """
    assert env is not None
    servo_module.set_and_check_cam(env.camera)

    is_sim = isinstance(env, RobotSimEnv)
    use_queue = True  # if False ignores action from the trajectory motion queue
    servo_action = None
    servo_control = None  # means default
    servo_queue = None
    done = False

    for counter in range(max_steps):
        state, reward, done, info = env.step(servo_action, servo_control)

        if done:
            break

        if is_sim:
            obs_image = state
        else:
            obs_image = info['rgb_unscaled']
        ee_pos = info['tcp_pose']
        servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
        servo_action, servo_done, servo_queue = servo_res

        if start_paused:
            servo_action, servo_done, servo_queue = None, None, None
            if servo_module.view_plots:
                start_paused = not servo_module.view_plots.started
            continue

        if use_queue and servo_queue:
            for _ in range(len(servo_queue)):
                name, val = servo_queue.pop(0)
                servo_action, servo_control = servo_module.cmd_to_action(env, name, val, servo_action)
                state, reward, done, info = env.step(servo_action, servo_control)
            continue

        if servo_module.config.mode == "pointcloud-abs":
            servo_action, servo_control = servo_module.abs_to_action(env, servo_action, info)
        else:
            servo_control = None  # means default


    if servo_module.view_plots:
        del servo_module.view_plots

    info['ep_length'] = counter

    return state, reward, done, info


def main_sim():
    """
    The main function that loads the recording, then runs policy.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    recording, episode_num = "./tmp_test/pick_n_place", 0
    control_config = dict(mode="pointcloud",
                          gain_xy=50,
                          gain_z=100,
                          gain_r=15,
                          threshold=0.40)  # .15 35 45

    # TODO(max): save and load these value from a file.
    task_name = "pick_n_place"
    robot = "kuka"
    renderer = "debug"
    control = "relative"

    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  control_config=control_config,
                                  plot=True, save_dir=None)

    env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer,
                      control=control, max_steps=500, show_workspace=False,
                      param_randomize=True, img_size=(256, 256))

    state, reward, done, info = evaluate_control(env, servo_module)

    print("reward:", reward, "\n")


def main_hw():
    from gym_grasping.envs.iiwa_env import IIWAEnv
    logging.basicConfig(level=logging.INFO, format="")

    recording, episode_num = "/media/argusm/Seagate Expansion Drive/kuka_recordings/flow/vacuum", 5
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/multi2", 1
    # recording, episode_num = "/media/kuka/sergio-ntfs/multi2/", 1

    control_config = dict(mode="pointcloud",
                          gain_xy=50,
                          gain_z=100,
                          gain_r=15,
                          threshold=0.35)

    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  control_config=control_config,
                                  plot=True, save_dir=None)

    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced',
                       img_flip_horizontal=True,
                       dv=0.0035, drot=0.025, use_impedance=True, max_steps=1e9,
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi/2), control='relative',
                       gripper_opening_width=109,
                       obs_dict=False)
    iiwa_env.reset()

    state, reward, done, info = evaluate_control(iiwa_env, servo_module)
    print("reward:", reward, "\n")


if __name__ == "__main__":
    # just to avoid having to set this
    node = platform.uname().node
    if node in ('plumbum', 'lurleen'):
        main_hw(start_paused=True)
    else:
        main_sim()
