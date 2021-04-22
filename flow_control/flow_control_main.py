"""
Testing file for development, to experiment with evironments.
"""
import time
import logging
from pdb import set_trace
import numpy as np
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule


def evaluate_control(env, recording, episode_num, start_index=0,
                     control_config=None, max_steps=1000, plot=True):
    """
    Function that runs the policy.
    """
    assert env is not None

    # load the servo module
    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  start_index=start_index,
                                  control_config=control_config,
                                  camera_calibration=env.camera.calibration,
                                  plot=plot, save_dir=None)
    do_abs = True
    servo_action = None
    servo_control = None  # means default
    servo_queue = None
    done = False
    for counter in range(max_steps):
        # environment stepping
        state, reward, done, info = env.step(servo_action, servo_control)
        if done:
            break

        if not servo_queue:
            # compute action
            if isinstance(env, RobotSimEnv):
                obs_image = state  # TODO(max): fix API change between sim and robot
            else:
                obs_image = info['rgb_unscaled']
            ee_pos = info['robot_state_full'][:8]  # take three position values
            servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
            servo_action, servo_done, servo_queue = servo_res
            servo_control = None  # means default

        if not do_abs or not servo_queue:
            continue

        # TODO(max): there should be only one call to env.step
        name, val = servo_queue.pop(0)
        servo_action, servo_control = servo_module.cmd_to_action(env, name, val, servo_action)
        state, reward, done, info = env.step(servo_action, servo_control)

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

    env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer,
                      control=control, max_steps=500, show_workspace=False,
                      param_randomize=True, img_size=(256, 256))

    state, reward, done, info = evaluate_control(env, recording,
                                                 episode_num=episode_num,
                                                 control_config=control_config,
                                                 plot=True)
    print("reward:", reward, "\n")


def main_hw():
    import math
    from gym_grasping.envs.iiwa_env import IIWAEnv
    logging.basicConfig(level=logging.INFO, format="")

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/vacuum", 5
    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/multi2", 1

    control_config = dict(mode="pointcloud",
                          gain_xy=50,
                          gain_z=100,
                          gain_r=15,
                          threshold=0.35)

    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced',
                       dv=0.0035, drot=0.025, use_impedance=True, max_steps=1e9,
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='relative',
                       # gripper_opening_width=109
                       )

    iiwa_env.reset()
    state, reward, done, info = evaluate_control(iiwa_env, recording,
                                                 episode_num=episode_num,
                                                 control_config=control_config,
                                                 plot=True)
    print("reward:", reward, "\n")


if __name__ == "__main__":
    # main_sim()
    main_hw()
