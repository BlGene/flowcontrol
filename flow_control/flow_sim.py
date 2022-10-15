"""
Testing file for development, to experiment with environments.
"""
import os
import logging

from gym_grasping.envs.robot_sim_env import RobotSimEnv

from flow_control.servoing.module import ServoingModule
from flow_control.runner import evaluate_control


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

    servo_module = ServoingModule(recording, control_config=control_config, plot=True, save_dir=plot_save)

    env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer,
                      control=control, max_steps=500, show_workspace=False,
                      param_randomize=True, img_size=(256, 256))

    _, reward, _, _ = evaluate_control(env, servo_module)

    print("reward:", reward, "\n")


if __name__ == "__main__":
    main_sim()
