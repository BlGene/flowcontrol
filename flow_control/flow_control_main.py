"""
Testing file for development, to experiment with environments.
"""
import os
import math
import logging
import platform
import time

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule


def evaluate_control(env, servo_module, start_paused=False, max_steps=1000):
    """
    Function that runs the policy.
    """
    assert env is not None
    servo_module.set_env(env)

    if start_paused:
        logging.info("Starting paused.")

    use_queue = True  # if False ignores action from the trajectory motion queue

    servo_action = None
    servo_control = None  # means env's default
    state = {'robot_state': None}
    for counter in range(max_steps):

        state, reward, done, info = env.step(servo_action, servo_control)
        if done:
            break

        # Normal servoing, based on correspondences
        servo_action, servo_done, servo_info = servo_module.step(state, info)
        servo_control = env.robot.get_control("relative")
        #assert len(servo_action) == 5
        if servo_done:
            break

        if start_paused:
            if servo_module.view_plots:
                start_paused = not servo_module.view_plots.started
            servo_action, servo_done, servo_queue = None, None, None
            continue

        # Trajectory actions, based on the trajectory of the demo; dead recckoning.
        # These are the big actions.
        servo_queue = servo_info["traj_acts"] if "traj_acts" in servo_info else None
        if use_queue and servo_queue:
            for _ in range(len(servo_queue)):
                name, val = servo_queue.pop(0)
                print(f"Trajectory action: {name} val: {val}")
                if env.robot.name == "panda":
                    servo_action, servo_control = servo_module.cmd_to_action(env, name, val, servo_action)
                    goal_pos, goal_quat, goal_g = servo_action
                    env.robot.move_cart_pos_abs_lin(goal_pos, goal_quat)
                    if goal_g == 1:
                        env.robot.open_gripper()
                    elif goal_g == -1:
                        env.robot.close_gripper(blocking=True)
                        time.sleep(.3)
                    else:
                        raise ValueError(f"Bad gripper action: {goal_g} must be 1,-1")
                else:
                    servo_action, servo_control = servo_module.cmd_to_action(env, name, val, servo_action)
                    state, reward, done, info = env.step(servo_action, servo_control)
                    state, reward, done, info = env.step(servo_action, servo_control)
            servo_action, servo_control = None, None
            continue

        if servo_module.config.mode == "pointcloud-abs":
            # do a direct application of action, bypass the env
            # TODO(max): this should probably be removed, I think this was added
            # for the panda robot.
            env.robot.move_cart_pos_abs_lin(servo_action[0:3], servo_action[3:7])
            servo_action, servo_control = None, None

    if servo_module.view_plots:
        del servo_module.view_plots
    info['ep_length'] = counter

    print(f"\nServoing completed with reward: {reward}, ran for {counter} steps.\n")

    return state, reward, done, info


def main_sim():
    """
    The main function that loads the recording, then runs policy.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    recording, episode_num = "./tmp_test/pick_n_place", 0
    control_config = dict(mode="pointcloud", threshold=0.30)  # .15 35 45

    # TODO(max): save and load these value from a file.
    task_name = "pick_n_place"
    robot = "kuka"
    renderer = "debug"
    control = "relative"
    plot_save = os.path.join(recording, "plot")

    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  control_config=control_config,
                                  plot=True, save_dir=plot_save)

    env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer,
                      control=control, max_steps=500, show_workspace=False,
                      param_randomize=True, img_size=(256, 256))

    _, reward, _, _ = evaluate_control(env, servo_module)

    print("reward:", reward, "\n")


def main_hw(start_paused=False):
    """
    The main function that loads the recording, then runs policy for the iiwa.
    """
    from gym_grasping.envs.iiwa_env import IIWAEnv
    logging.basicConfig(level=logging.INFO, format="")

    recording, episode_num = "/media/argusm/Seagate Expansion Drive/kuka_recordings/flow/vacuum", 5
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/multi2", 1
    # recording, episode_num = "/media/kuka/sergio-ntfs/multi2/", 1

    control_config = dict(mode="pointcloud-abs", threshold=0.35)

    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  control_config=control_config,
                                  plot=True, save_dir=None)

    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced',
                       img_flip_horizontal=True,
                       dv=0.0035, drot=0.025, use_impedance=True, max_steps=1e9,
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='relative',
                       gripper_opening_width=109,
                       obs_dict=False)
    iiwa_env.reset()

    _, reward, _, _ = evaluate_control(iiwa_env, servo_module, start_paused=start_paused)
    print("reward:", reward, "\n")


if __name__ == "__main__":
    # just to avoid having to set this
    node = platform.uname().node
    if node in ('plumbum', 'lurleen'):
        main_hw(start_paused=True)
    else:
        main_sim()
