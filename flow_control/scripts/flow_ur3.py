"""
Module for testing UR3.
"""
import logging
import numpy as np

import hydra.utils

from robot_io.recorder.simple_recorder import SimpleRecorder

from flow_control.servoing.module import ServoingModule
from flow_control.servoing.runner import evaluate_control


log = logging.getLogger(__name__)


def move_to_neutral_desk(robot):
    home_pos = np.array([0.2988764, 0.11044048, 0.15792169])
    home_orn = np.array([9.99999242e-01, -1.23099822e-03, 1.18825773e-05, 3.06933556e-05])
    pos, orn = robot.get_tcp_pos_orn()

    if pos[2] > home_pos[2]:
        # current pos higher than home pos, add minimal delta_z
        delta_z = 0.02
    else:
        delta_z = abs(pos[2] - home_pos[2])

    # delta_z = max(0.0, pos[2] - home_pos[2])
    pos_up = pos + np.array([0.0, 0.0, delta_z])
    hpos = np.array([0.2988764, 0.11044048, pos_up[2]])
    pos_up2 = hpos

    robot.move_cart_pos(pos_up, orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(pos_up2, home_orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(home_pos, home_orn, ref="abs", blocking=True, path="lin")


@hydra.main(config_path="/conf", config_name="ur3_teleop.yaml")
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
    # task = '/home/argusm/Desktop/Demonstrations/2023-01-18/18-27-16/'

    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(task, control_config=control_config, start_paused=True,
                                  plot=True, plot_save_dir='./plots')

    simp_rec = SimpleRecorder(env)
    state, _, _, _ = evaluate_control(env, servo_module, recorder=simp_rec)

    move_to_neutral_desk(robot)


if __name__ == "__main__":
    main()
