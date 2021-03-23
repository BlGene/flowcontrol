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
            if do_abs is False:
                servo_queue = None

        if not servo_queue:
            continue

        name, val = servo_queue.pop(0)
        if name == "grip":
            servo_control = env.robot.get_control("absolute", min_iter=24)
            pos = env.robot.desired_ee_pos
            rot = env.robot.desired_ee_angle
            servo_action = [*pos, rot, val]
            state, reward, done, info = env.step(servo_action, servo_control)

        elif name == "abs":
            servo_control = env.robot.get_control("absolute")
            rot = env.robot.desired_ee_angle
            servo_action = [*val[0:3], rot, servo_action[-1]]
            state, reward, done, info = env.step(servo_action, servo_control)

        elif name == "rel":
            servo_control = env.robot.get_control("absolute")
            new_pos = np.array(env.robot.get_tcp_pos()) + val[0:3]
            rot = env.robot.desired_ee_angle
            servo_action = [*new_pos, rot, servo_action[-1]]
            state, reward, done, info = env.step(servo_action, servo_control)
        else:
            raise ValueError

        # TODO(max): replace the env.step calls with setting actions
        # then move cmd_to_action code to servoing module as staticmethod

    if servo_module.view_plots:
        del servo_module.view_plots
    info['ep_length'] = counter

    return state, reward, done, info


def main():
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


if __name__ == "__main__":
    main()
