"""
Module for testing UR3.
"""
import os
import logging

import hydra.utils

from robot_io.recorder.simple_recorder import SimpleRecorder

from flow_control.servoing.module import ServoingModule
from flow_control.runner import evaluate_control


log = logging.getLogger(__name__)


def move_to_neutral_desk(robot):
    """
    Move the robot to a neutral position, by first moving along the z-axis only, then the rest.
    """
    robot.open_gripper()

    neutral_pos, neutral_orn = (0.30, 0.11, 0.16), (1, 0, 0, 0)  # queried from move_to_neutral()
    cur_pos, _ = robot.get_tcp_pos_orn()

    # return home high first
    # pos_up = np.array([0, 0, cur_pos[2] - neutral_pos[2]])
    # orn_up = np.array([0, 0, 0, 1])
    # robot.move_cart_pos(pos_up, orn_up, ref="rel", path="ptp")

    robot.move_cart_pos(neutral_pos, neutral_orn, ref="abs", path="ptp")


@hydra.main(config_path="/home/argusm/lang/robot_io/conf", config_name="ur3_teleop.yaml")
def main(cfg):
    """
    Try running conditional servoing.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # return to neutral in a safer way.
    move_to_neutral_desk(robot)

    # task = '/home/argusm/lmb/robot_recordings/flow/sick_wtt/16-41-43'
    task = '/home/argusm/lmb/robot_recordings/flow/sick_wtt/16-51-30'

    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(task, control_config=control_config, start_paused=True,
                                  plot=True, plot_save_dir='./plots')

    simp_rec = SimpleRecorder(env)
    state, _, _, _ = evaluate_control(env, servo_module, recorder=simp_rec)


if __name__ == "__main__":
    main()
